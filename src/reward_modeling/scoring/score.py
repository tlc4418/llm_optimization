import math
import torch
from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from typing import Union

from alpaca_farm.models.reward_model import RewardModel
from accelerate import Accelerator, DistributedType
from src.data_utils.rm_dataset_formatter import RMPromptDataset
from model_training.models.reward_model import (
    GPTNeoXRewardModel,
    GPTNeoXRewardModelConfig,
)

MAX_LEN = 776  # 520 instruction + 256 answer


def get_reward(
    samples,
    reward_models,
    reward_tokenizer,
    reward_device,  # needed?
    batch_size,
    objective_function=None,
    weight=None,
    is_alpacafarm_rm=False,
):
    if not isinstance(reward_models, list):
        reward_models = [reward_models]

    input = reward_tokenizer(
        samples,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(reward_device)

    all_rewards = []
    for reward_model in reward_models:
        out = []
        for i in range(math.ceil(len(samples) / batch_size)):
            batch_ixs = slice(i * batch_size, (i + 1) * batch_size)
            input_ids = input.input_ids[batch_ixs]
            attention_mask = input.attention_mask[batch_ixs]
            output = reward_model(input_ids, attention_mask)
            rewards = output.rewards if is_alpacafarm_rm else output.logits[:, 0]
            out.extend(rewards)
        all_rewards.append(torch.hstack(out))

    if len(all_rewards) == 1:
        all_rewards = all_rewards[0]
        return all_rewards, torch.empty_like(all_rewards)

    all_rewards = torch.stack(all_rewards, 0)
    var = torch.var(all_rewards, dim=0)
    if objective_function:
        all_rewards = objective_function(all_rewards, weight)
    return all_rewards, var


def score_answers(
    model_name: str,
    dataset: Union[str, Dataset],
    split: str = "validation",
    scores_type: str = "gold_scores",
    sort: bool = False,
    split_size: int = 32,
    is_alpacafarm_rm: bool = False,
) -> list:
    dataset = load_dataset(dataset)[split] if isinstance(dataset, str) else dataset
    prompt_dataset = RMPromptDataset(
        dataset,
        output_alpaca=is_alpacafarm_rm,
    )
    model = (
        RewardModel.from_pretrained(model_name, flash_attn=True, bf16=True)
        if is_alpacafarm_rm
        else AutoModelForSequenceClassification.from_pretrained(model_name)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    accelerator = Accelerator()

    # Hacky, maybe look for better option.
    # But enables PPO gold evaluation to run after training.
    # The other option is to run gold score evaluation separately from PPO training.
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["stage"] = 0

    model, tokenizer = accelerator.prepare(model, tokenizer)
    model.eval()
    model.requires_grad_(False)

    samples = [prompts for _, prompts in prompt_dataset]
    has_multi_answers = len(samples[0]) > 1

    # If each prompt has multiple answers, create inter-prompt batches
    if has_multi_answers:
        rewards = [
            get_reward(
                prompts,
                model,
                tokenizer,
                model.device,
                split_size,
                is_alpacafarm_rm=is_alpacafarm_rm,
            )[0]
            for prompts in samples
        ]

    # Otherwise create intra-prompt batches
    else:
        rewards, _ = get_reward(
            [prompts[0] for prompts in samples],
            model,
            tokenizer,
            model.device,
            split_size,
            is_alpacafarm_rm=is_alpacafarm_rm,
        )

    data = []
    for i, (entry, _) in enumerate(prompt_dataset):
        scores = rewards[i].cpu().detach()

        if has_multi_answers:
            if sort:
                scores, indices = torch.sort(scores)
                entry["answers"] = [entry["answers"][i] for i in indices]
                if entry.get("gold_scores"):
                    entry["gold_scores"] = [entry["gold_scores"][i] for i in indices]
                if entry.get("proxy_scores"):
                    entry["proxy_scores"] = [entry["proxy_scores"][i] for i in indices]

        entry[scores_type] = scores.tolist() if has_multi_answers else [scores.item()]
        data.append(entry)

    return data


AutoConfig.register("gpt_neox_reward_model", GPTNeoXRewardModelConfig)
AutoModelForSequenceClassification.register(GPTNeoXRewardModelConfig, GPTNeoXRewardModel)
