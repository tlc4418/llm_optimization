import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.reward_modeling.scoring.score import get_reward


# conservative optimization objectives -------------------------------------------------


def mean_objective_function(all_scores, coeff=None):
    return torch.mean(all_scores, dim=0)


def worst_case_optimization(all_scores, coeff=None):
    return torch.min(all_scores, dim=0)[0]


def uncertainty_weighted_optimization(all_scores, coeff=1):
    return torch.mean(all_scores, dim=0) - coeff * torch.var(all_scores, dim=0)


OBJECTIVE_FUNCTIONS = {
    "mean": mean_objective_function,
    "WCO": worst_case_optimization,
    "UWO": uncertainty_weighted_optimization,
}

# --------------------------------------------------------------------------------------


def create_reward_fn(model_names: list[str], objective_name: str = "mean", weight=None):
    batch_size = 32
    reward_device = torch.cuda.device_count() - 1
    print(f"Using device {reward_device} for reward model", flush=True)

    reward_tokenizer = AutoTokenizer.from_pretrained(model_names[0])

    reward_models = []
    for model_name in model_names:
        reward_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        reward_model.eval()
        reward_model.requires_grad_(False)
        reward_model.to(reward_device)
        reward_models.append(reward_model)

    objective_function = OBJECTIVE_FUNCTIONS[objective_name]

    def format_samples(prompts, outputs, eval):
        return [
            f"<|prompter|>{prompt}<|endoftext|><|assistant|>{output}{'<|endoftext|>' if eval else ''}"
            for prompt, output in zip(prompts, outputs)
        ]

    def reward_fn(samples, prompts, outputs, eval=False, **kwargs):
        samples = format_samples(prompts, outputs, eval)
        rewards, vars = get_reward(
            samples,
            reward_models,
            reward_tokenizer,
            reward_device,
            batch_size,
            objective_function,
            weight,
        )
        return rewards, vars

    return reward_fn
