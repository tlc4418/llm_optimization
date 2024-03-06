# This file builds on Open-Assistant's model_training/custom_datasets/__init__.py and
# model_training/utils/utils.py, but enables the retrieval of custom datasets and
# loaders from src/data_utils/oacustom_datasets/rank_datasets.py and
# src/data_utils/oacustom_datasets/dataset_loader.py.

from typing import Optional

from torch.utils.data import ConcatDataset, Dataset, Subset
from model_training.custom_datasets import get_one_dataset
from model_training.utils.utils import get_dataset_name_and_kwargs_from_data_config

from src.data_utils.oa_custom_datasets.dataset_loader import (
    CUSTOM_RL_DATASETS,
    CUSTOM_RM_DATASETS,
    CUSTOM_SFT_DATASETS,
    load_custom_dataset,
)


def get_dataset(
    conf,
    mode: str = "sft",
) -> tuple[ConcatDataset, dict[str, Subset]]:
    train_datasets, evals = [], {}

    for data_config in conf.datasets + conf.datasets_extra:
        dataset_name, kwargs = get_dataset_name_and_kwargs_from_data_config(data_config)
        train, val = custom_get_one_dataset(conf, dataset_name, mode=mode, **kwargs)
        train_datasets.append(train)

        if val is not None:
            evals[dataset_name] = Subset(val, list(range(min(len(val), conf.eval_size)))) if conf.eval_size else val

    train = ConcatDataset(train_datasets)

    return train, evals


def custom_get_one_dataset(
    conf,
    dataset_name: str,
    val_split: float = 0.2,
    data_path: str = None,
    mode: str = "sft",
    max_val_set: Optional[int] = None,
    **kwargs,
) -> tuple[Dataset, Dataset | None]:
    try:
        return get_one_dataset(conf, dataset_name, val_split, data_path, mode, max_val_set, **kwargs)

    # If the dataset is not in the default datasets, try to load it as a custom dataset
    except (AssertionError, ValueError) as e:
        print(f"Failed to load dataset {dataset_name} as a default dataset: {e}")
        if mode == "rl":
            assert dataset_name in CUSTOM_RL_DATASETS, f"Dataset {dataset_name} not supported for RL"

        if mode == "rm":
            assert dataset_name in CUSTOM_RM_DATASETS, f"Dataset {dataset_name} not supported for reward modeling"

        if any(dataset_name in x for x in [CUSTOM_SFT_DATASETS, CUSTOM_RM_DATASETS, CUSTOM_RL_DATASETS]):
            print(f"Loading custom dataset {dataset_name}")
            return load_custom_dataset(dataset_name, mode, **kwargs)
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")
