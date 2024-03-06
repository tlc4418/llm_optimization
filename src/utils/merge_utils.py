import numpy as np
from datasets import concatenate_datasets, load_dataset


def merge_splits(filename_format, splits):
    """Merge dataset splits into a single dataset"""

    all_split_data = [
        load_dataset(
            "json",
            data_files=filename_format.format(split=i),
        )["train"]
        for i in splits
    ]
    dataset = concatenate_datasets(all_split_data)
    return dataset


def combine_bon(all_data, add_std=False, add_avg_answer_length=False):
    """Combine BoN results from different files"""

    combined_data = {}

    combined_data["n"] = all_data[0]["n"]
    combined_data["proxy_score"] = list(np.mean([data["proxy_score"] for data in all_data], axis=0))
    combined_data["gold_score"] = list(np.mean([data["gold_score"] for data in all_data], axis=0))

    if add_avg_answer_length:
        combined_data["avg_answer_length"] = list(np.mean([data["avg_answer_length"] for data in all_data], axis=0))
        combined_data["std_answer_length"] = list(np.std([data["avg_answer_length"] for data in all_data], axis=0))

    if add_std:
        combined_data["proxy_score_std"] = list(np.std([data["proxy_score"] for data in all_data], axis=0))
        combined_data["gold_score_std"] = list(np.std([data["gold_score"] for data in all_data], axis=0))

    return combined_data


def get_avg_scores(dataset):
    all_proxy_scores = dataset["proxy_scores"]
    all_gold_scores = dataset["gold_scores"]

    proxy_mean_offset = np.mean(all_proxy_scores)
    proxy_var_offset = np.var(all_proxy_scores)
    gold_mean_offset = np.mean(all_gold_scores)
    gold_var_offset = np.var(all_gold_scores)

    return proxy_mean_offset, proxy_var_offset, gold_mean_offset, gold_var_offset
