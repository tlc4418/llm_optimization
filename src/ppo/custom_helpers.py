import json
import typer

from datasets import load_dataset
from pathlib import Path
from typing_extensions import Annotated

from src.reward_modeling.scoring.ppo_reward_functions import create_reward_fn
from src.reward_modeling.scoring.score import score_answers


def process_configs(training_conf, rank_config, trlx_config):
    """Process the training and rank configs and set the output directory"""

    trlx_config.train.seed = training_conf.rng_seed
    output_dir = training_conf.output_dir

    if training_conf.rm_seed is not None:
        output_dir = f"{output_dir}/seed{training_conf.rm_seed}"
    elif len(rank_config.model_names) > 1:
        type = rank_config.objective_name
        weight_str = f"-{rank_config.uwo_weight}" if rank_config.uwo_weight else ""
        output_dir = f"{output_dir}/ensemble_{type}{weight_str}"
    else:
        output_dir = f"{output_dir}/{rank_config.model_names[0].replace('/', '_')}"

    trlx_config.train.output_dir = output_dir
    trlx_config.train.run_name = output_dir

    Path(f"{output_dir}/eval").mkdir(parents=True, exist_ok=True)

    return output_dir


def get_reward_fn(rank_config, training_conf):
    """Get the reward function for PPO training."""

    if len(rank_config.model_names) > 1:
        print(f"Using ensemble reward function {rank_config.objective_name}")
        reward_fn = create_reward_fn(rank_config.model_names, rank_config.objective_name, rank_config.uwo_weight)
    else:
        model_name = rank_config.model_names[0]
        if training_conf.rm_seed is not None:
            model_name = model_name.format(seed=training_conf.rm_seed)
        reward_fn = create_reward_fn([model_name])

    return reward_fn


def gold_score(
    eval_dir: Annotated[
        str,
        typer.Argument(
            help="Eval directory containing proxy evaluation result files "
            "for each evaluation step. Gold scores will be added to these "
            "results."
        ),
    ],
    gold_rm_model_name: Annotated[str, typer.Argument(help="Name of (or path to) the gold RM model.")],
    is_alpacafarm_rm: Annotated[bool, typer.Option(help="Whether the RM is from AlpacaFarm.")] = False,
    batch_size: Annotated[int, typer.Option(help="The batch size for scoring with the gold RM.")] = 32,
):
    """Score the PPO evaluations with the gold RM."""

    eval_files = [f for f in Path(f"{eval_dir}").iterdir() if f.is_file()]

    for eval_file in eval_files:
        dataset = load_dataset("json", data_files=str(eval_file))["train"]
        scored_data = score_answers(
            model_name=gold_rm_model_name,
            dataset=dataset,
            scores_type="gold_scores",
            sort=False,
            split_size=batch_size,
            is_alpacafarm_rm=is_alpacafarm_rm,
        )
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(scored_data, f, ensure_ascii=False, indent=4)
