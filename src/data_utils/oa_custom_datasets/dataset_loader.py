from torch import Generator
from torch.utils.data import Dataset, Subset, random_split
from datasets import load_dataset
from src.data_utils.oa_custom_datasets.rank_datasets import (
    AlpacaFarmHumanPref,
    CustomHFPref,
)
from model_training.custom_datasets.formatting import (
    create_dataset_entry_qa,
    DatasetEntry,
)
from model_training.custom_datasets.qa_datasets import (
    AlpacaBaseDataset,
)
from model_training.custom_datasets.utils import (
    _filter_by_words,
)


def load_custom_dataset(dataset_name: str, mode: str, **kwargs) -> tuple[Dataset, Dataset]:
    """
    Loads a custom dataset, ready to be used in the Open-Assistant training pipeline.
    """
    print(f"Loading custom dataset {dataset_name} for mode {mode}", flush=True)

    if mode == "sft":
        assert dataset_name in CUSTOM_SFT_DATASETS, f"Dataset {dataset_name} not supported for supervised fine-tuning"
        datasets = CUSTOM_SFT_DATASETS
    elif mode == "rm":
        assert dataset_name in CUSTOM_RM_DATASETS, f"Dataset {dataset_name} not supported for reward modeling"
        datasets = CUSTOM_RM_DATASETS
    elif mode == "rl":
        assert dataset_name in CUSTOM_RL_DATASETS, f"Dataset {dataset_name} not supported for RL"
        datasets = CUSTOM_RL_DATASETS
    else:
        raise ValueError(f"Mode {mode} not supported")

    kwargs["mode"] = mode
    return datasets[dataset_name](**kwargs)


def load_alpaca_dataset(
    mode: str = "sft",
    dataset_path: str = "tatsu-lab/alpaca_farm",
    **kwargs,
) -> tuple[AlpacaBaseDataset, AlpacaBaseDataset]:
    """
    Taken and modified from Open-Assistant's model/model_training/
    custom_datasets/qa_datasets.py load_alpaca_dataset()

    Loads the AlpacaFarm QA dataset for the specified mode.
    """

    def process_split(dataset: Subset) -> list[DatasetEntry]:
        data = []
        for row in dataset:
            question = row["instruction"]
            if len(row["input"]) > 0:
                input_ = "{}\n{}".format(question, row["input"])
            else:
                input_ = question

            if (_filter_by_words(input_) is None) or (_filter_by_words(row["output"]) is None):
                continue

            ds_entry = (
                create_dataset_entry_qa(mode=mode, questions=[input_], answers=[row["output"]])
                if mode == "sft"
                else [input_]
            )
            data.append(ds_entry)
        return data

    dataset = load_dataset(dataset_path, "alpaca_instructions")
    train = process_split(dataset["sft"]) if mode == "sft" else process_split(dataset["unlabeled"])
    validation = process_split(dataset["val"])
    return train, validation


def load_alpacafarm_human_pref(
    eval_size: int = 500,
    manual_seed: int = 287631038922,
    **kwargs,
) -> tuple[Dataset, Dataset]:
    """Loads the AlpacaFarm Human Preference dataset into the expected format."""

    generator = Generator()
    generator.manual_seed(manual_seed)

    dataset = load_dataset("tatsu-lab/alpaca_farm", "alpaca_human_preference")["preference"]
    new_train_size = len(dataset) - eval_size
    train_dataset, eval_dataset = random_split(dataset, [new_train_size, eval_size], generator=generator)
    train = AlpacaFarmHumanPref(train_dataset)
    validation = AlpacaFarmHumanPref(eval_dataset)
    return train, validation


def load_custom_hf_pref(
    dataset_path: str = "tlc4418/1.4b-policy_preference_data_gold_labelled",
    train_size: int = 46000,
    eval_size: int = 2000,
    **kwargs,
) -> tuple[Dataset, Dataset]:
    """
    Loads a custom HuggingFace preference dataset into the expected format.

    Here the dataset is 'tlc4418/1.4b-policy_preference_data_gold_labelled', but any
    preference dataset can be used as long as it follows the format of this provided
    dataset, and has train/validation splits.
    """

    train_dataset, eval_dataset = load_dataset(
        dataset_path,
        split=["train", "validation"],
    )
    train = CustomHFPref(train_dataset, train_size)
    validation = CustomHFPref(eval_dataset, eval_size, train=False)
    return train, validation


# Map of custom datasets to their respective loaders -----------------------------------

CUSTOM_SFT_DATASETS = {
    "alpaca_farm": load_alpaca_dataset,
}

CUSTOM_RM_DATASETS = {
    "alpaca_farm_pref": load_alpacafarm_human_pref,
    "custom_hf_pref": load_custom_hf_pref,
}

CUSTOM_RL_DATASETS = {
    "alpaca_farm": load_alpaca_dataset,
}
