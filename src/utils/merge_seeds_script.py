from datasets import load_dataset
import json
import math
from pathlib import Path

from sft import combine_bon, get_avg_scores, merge_splits


if __name__ == "__main__":
    splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seeds = [1, 2, 3, 4, 5]

    bon_filename = "runs/1.4b-policy_44m-proxy_20k_seed{seed}/bon_sampled_results/split-{split}.json"
    labelled_filename = "runs/1.4b-policy_44m-proxy_20k_seed{seed}/proxy_labelled_sorted/split-{split}.json"

    seed_data = []
    for s in seeds:
        all_split_data = [
            load_dataset(
                "json",
                data_files=bon_filename.format(split=i, seed=s),
            )["train"]
            for i in splits
        ]
        bon_dataset = combine_bon(all_split_data, add_avg_answer_length=True)

        # Process and normalize scores
        dataset = merge_splits(labelled_filename.format(split="{split}", seed=s), splits)
        (
            proxy_mean_offset,
            proxy_var_offset,
            gold_mean_offset,
            gold_var_offset,
        ) = get_avg_scores(dataset)
        bon_dataset["proxy_score"] = (bon_dataset["proxy_score"] - proxy_mean_offset) / math.sqrt(proxy_var_offset)
        bon_dataset["gold_score"] = (bon_dataset["gold_score"] - gold_mean_offset) / math.sqrt(gold_var_offset)

        seed_data.append(bon_dataset)

    combined_seeds = combine_bon(seed_data, add_std=True, add_avg_answer_length=True)

    new_bon_sample_folder = (
        f"sft/runs/1.4b-policy_44m-proxy_20k_seeds{('-'.join(list(map(str, seeds))))}/bon_sampled_results"
    )
    Path(new_bon_sample_folder).mkdir(parents=True, exist_ok=True)

    with open(new_bon_sample_folder + "/splits-merged.json", "w", encoding="utf-8") as f:
        json.dump(combined_seeds, f, ensure_ascii=False, indent=4)
