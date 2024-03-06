# Reward model training
An explanation for [running reward model training](#running-reward-model-training) is provided in the project-level README, but we also provide it here for convenience.


# Running reward model training
Very similar to SFT, you can perform reward model training on any registered HuggingFace or local reward model by creating a new entry in the [RM config file](/configs/config_rm.yaml). The following are crucial fields:
| Field name   | Type | Description | Example value |
| ------------ | ---- | ----------- | ------------- |
| `model_name` | str  | name of (path to) the model to be trained. Normal Pythia GPTNeoX models will be automatically converted into a reward model in this training process, such that outputs from the previous SFT training step can be directly fed in. | models/pythia_model_70m_sft |
| `output_dir` | str  | name of (path to) the directory where the output model should be saved. "_seed{rng_seed}" will get appended to it. | models/rm-pythia-44m |
| `datasets`   | list | list of preference datasets to use for training (see [dataset guide](/src/data_utils/oa_custom_datasets/README.md) for details) | - alpaca_farm_pref |
| `rng_seed`   | int  | seed which controls the RM training (RM head initialisation and dataset order). This is very useful to create different RMs for a reward model ensemble. It can also be set as a command-line option (see below). | 1 |

Again, default hyperparameters can be overwritten.

Reward model training can then be started with this new config entry (e.g. "rm-pythia-44m"). The `--rng_seed` argument is optional and will otherwise be sourced from the RM config. Command to launch training:
```
accelerate launch --config_file configs/accelerate_config.yaml src/reward_modeling/training/trainer_rm.py --configs defaults_rm {your_rm_config_entry} --rng_seed {your_choice_seed}
```
