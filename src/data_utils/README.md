# Training datasets (and BoN dataset)

Datasets are needed at every step of the training pipeline:
- Instruction-tuning QA datasets for SFT
- Preference datasets for reward model training
- More QA datasets for RL
- Special case for BoN sampling

As we use Open-Assistant for our training pipeline, we need to ensure our datasets are compatible with this workflow. In our [paper](https://arxiv.org/abs/2310.02743), we use AlpacaFarm-based datasets throughout our experiments (note that for RM training we use the dataset presented in the [last section](#example-custom-preference-dataset-huggingface)). We provide functionality for using these datasets at all stages of training and demonstrate how to easily extend the current datasets to use your own custom datasets.

## Provided AlpacaFarm-based datasets
By impementing the right classes and wrappers, we make it very straightforward to use AlpacaFarm-based datasets for your training.

### For SFT
Simply add 'alpaca_farm' in the `datasets` section of the [SFT config](/configs/config.yaml#L101). Your model will then be trained using the `sft` split of the AlpacaFArm instruction dataset.

### For RM training
We adapt the AlpacaFarm human preference dataset to fit the Open-Assistant flow, such that you can simply add 'alpaca_farm_pref' to the `datasets` section of the [RM config](/configs/config_rm.yaml#53), as for SFT. Your model will then be trained using the `alpaca_human_preference` split of the AlpacaFArm instruction dataset.

### For RL training
Again, for RL simply add 'alpaca_farm' in the `datasets` section of the [RL config](/configs/config_rl.yaml#L18). Your model will then be trained using the `unlabeled` split of the AlpacaFarm instruction dataset.


## Using your own datasets
You can also use your own datasets for these training stages. This is done by creating a dataset loader in [dataset_loader.py](/src/data_utils/oa_custom_datasets/dataset_loader.py), and then adding the dataset name and pointer to the loader at the bottom of the file (in the appropriate dataset dictionary):
```python
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
```

Once these have been set, these can simply be added to the corresponding configs as seen in the AlpacaFArm-based example above. Any additional parameters defined can also be passed along (e.g. the number of training/eval samples as shown [below](#example-custom-preference-dataset-huggingface)).

### Format to follow
There is a specific format your instruction datasets need to follow. 

Firstly, the prompt text must be surrounded by `<|prompter|>`and `<|endoftext|>` tags, e.g '<|prompter|>Input<|endoftext|>'. The full text must be included within these tags (in the case of the AlpacaFarm dataset this means combining the `Instruction` and `Input` columns)

Similarly, the 'answer' text must be surrounded by `<|assistant|>` and `<|endoftext|>`, e.g. '<|assistant|>Answer<|endoftext|>'.

We then look at the structure of a dataset entry.

#### SFT
Each entry needs to be list of alternating 'Questions' (or instructions) and 'Answers', with the format presented above. In the AlpacaFarm example, these entries are simply pairs `[question, answer]`.

#### Reward modelling
For reward modelling, you need a preference dataset. Each entry should be a tuple of prompt and answers to the prompt, ranked in order of preference (best answer first). The prompt can be a list to support a conversation thread as a prompt, but in the AlpacaFarm this would be a singleton prompt, resulting in a dataset entry will the following format: `([prompt], [preferred_answer, worse_answer])`.

#### RL (PPO)
RL datasets have the same structure as for [SFT](#sft-and-rl), and as such the same format applies.

## Example custom preference dataset (HuggingFace)

We provide an example showing how to use a custom preference dataset for RM training. Any preference dataset stored in the HuggingFace hub can be used, but here we use `tlc4418/1.4b-policy_preference_data_gold_labelled`, which is used for RM training in our paper. It uses prompts from the AlpacaFarm dataset, but answers are generated form our initial 1.4B Pythia policy model (after SFT), and these are labelled for preferences using the AlpacaFarm 7B human preference reward model (see paper for more details).

Creating a dataset wrapper and loader in [rank_datasets.py](/src/data_utils/oa_custom_datasets/rank_datasets.py) and [dataset_loader.py](/src/data_utils/oa_custom_datasets/dataset_loader.py) respectively, we can simply use the 'custom_hf_pref' dataset in the [RM config](/configs/config_rm.yaml#53) and add the path to the desired HuggingFace dataset as shown. You can also define how many samples to use from each of the training an eval splits. Any dataset can be used so long as it contains the following columns: 

| instruction (str) | input (str) | answers (list) | preference (int) |
| ------------- | ------------- | ------------- | ------------- |
| Instruction to be used as a prompt to the LLM  | Optional input providing additional information for the prompt, often an input for the given instruction.  | List of 2 answers to the prompt (instruction + input) | 0 or 1, indicating the index of the preferred answer |

## Best-of-*n* (BoN)
BoN does not follow the Open-Assistant training pipeline, and as such has slightly different dataset rules. Each dataset entry is expected to have the follwoing columns:
| instruction (str) | input (str) | answers (list) | gold_scores (int) |
| ------------- | ------------- | ------------- | ------------- |
| Instruction to be used as a prompt to the LLM  | Optional input providing additional information for the prompt, often an input for the given instruction.  | List of N answers to the prompt (instruction + input). N must be greater than the *n* desired for best-of-*n*, to work with the BoN unbiased estimator (see paper for details). | List of reward model scores (one for each answer) according to the "gold-standard" reward model used (we use a synthetic setup - see paper). |

