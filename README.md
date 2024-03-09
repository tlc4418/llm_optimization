# LLM Overoptimization and Reward Model Ensembles
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


This repository contains the code related to "[Reward Model Ensembles Help Mitigate Overoptimization](https://arxiv.org/abs/2310.02743)". In particular, it provides the following:
- An easy way to perform LLM instruction supervised fine-tuning (SFT) as laid out in the paper
- An easy way to create and train one or more reward models to be used in reward model ensembles (or on their own)
- A best-of-*n* inference pipeline, both with individual reward models and ensembles
- A PPO-based RLHF training pipeline, both for individual reward models and ensembles
- Some crucial models and datasets to facilitate future work and experiments akin to those in the paper

We hope you can find the code, models, and datasets provided helpful for your own research. 

Note: SFT, RM, and PPO training backends are based on the [Open-Assistant](https://github.com/LAION-AI/Open-Assistant) and [trlx](https://github.com/CarperAI/trlx) libraries. Base models are taken from the open-source [Pythia](https://github.com/EleutherAI/pythia) suite.


## Installation
```
git clone https://github.com/tlc4418/llm_optimization.git
cd llm_optimization
pip install -e .
```

## Provided models and datasets
We provide the following models and datasets on HuggingFace to promote and faciliatte reproducing experiments and future work.

### Models
- [tlc4418/pythia_1.4b_sft_policy](https://huggingface.co/tlc4418/pythia_1.4b_sft_policy): 1.4B Pythia model after SFT on the AlpacaFarm "sft" split. Used as the initial policy model in our experiments.
- [tlc4418/pythia_70m_sft](https://huggingface.co/tlc4418/pythia_70m_sft): 70M Pythia model after SFT on the AlpacaFarm "sft" split. Used as the base model for most of our reward model experiments (before RM training). Multiple reward models can be created from this during [reward model training](#reward-model-training) and it is relatively inexpensive to do so.

### Datasets
- [tlc4418/1.4b-policy_preference_data_gold_labelled](https://huggingface.co/datasets/tlc4418/1.4b-policy_preference_data_gold_labelled): Preference dataset using labels from the AlpacaFarm dataset, generated answers from a 1.4b fine-tuned Pythia policy model, and labelled using the AlpacaFarm "reward-model-human" as a gold reward model. Used to train reward models.
- [tlc4418/gold_labelled_gens](https://huggingface.co/datasets/tlc4418/gold_labelled_gens): Dataset of 12600 answer generations from the 1.4B Pythia SFT model (provided above), using the AlpacaFarm dataset "val" split, and labelled with the AlpacaFarm "reward-model-human" to give "gold" scores. Used for best-of-*n* inference. This dataset is particularly expensive to geenrate and we hope it can help other with future work.
- We provide wrappers and functionality for using these datasets, as well as different parts of the [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm) dataset. See the [dataset guide](/src/data_utils/README.md) for details on this and how to use your own datasets.


## Supervised fine-tuning (SFT)
You can easily perform SFT on any HuggingFace or local language models by creating a new entry in the [SFT config](/configs/config.yaml). An example is given for a 70M Pythia model. The following are crucial fields:
| Field name   | Type | Description | Example value |
| ------------ | ---- | ----------- | ------------- |
| `model_name` | str  | name of (path to) the model to be trained | EleutherAI/pythia-70m |
| `output_dir` | str  | name of (path to) the directory where the output model should be saved | models/pythia_model_70m_sft |
| `datasets`   | list | list of instruction datasets to use for training (see [dataset guide](/src/data_utils/README.md) for details) | - alpaca_farm |

Any default hyperparameters can also be overwritten. The 70M Pythia model sample config entry shows a few examples.

Once the config has been set, training can be started with the following command, using the new entry name you created (e.g. "pythia-70m"):
```
accelerate launch --config_file configs/accelerate_config.yaml src/sft/trainer_sft.py --configs defaults {your_sft_config_entry}
```


## Reward model training
Very similar to SFT, you can perform reward model training on any registered HuggingFace or local reward model by creating a new entry in the [RM config](/configs/config_rm.yaml). The following are crucial fields:
| Field name   | Type | Description | Example value |
| ------------ | ---- | ----------- | ------------- |
| `model_name` | str  | name of (path to) the model to be trained. Normal Pythia GPTNeoX models will be automatically converted into a reward model in this training process, such that outputs from the previous SFT training step can be directly fed in. | models/pythia_model_70m_sft |
| `output_dir` | str  | name of (path to) the directory where the output model should be saved. "_seed{rng_seed}" will get appended to it. | models/rm-pythia-44m |
| `datasets`   | list | list of preference datasets to use for training (see [dataset guide](/src/data_utils/README.md) for details) | - alpaca_farm_pref |
| `rng_seed`   | int  | seed which controls the RM training (RM head initialisation and dataset order). This is very useful to create different RMs for a reward model ensemble. It can also be set as a command-line option (see below). | 1 |

Again, default hyperparameters can be overwritten.

Reward model training can then be started with this new config entry (e.g. "rm-pythia-44m"). The `--rng_seed` argument is optional and will otherwise be sourced from the RM config. Command to launch training:
```
accelerate launch --config_file configs/accelerate_config.yaml src/reward_modeling/training/trainer_rm.py --configs defaults_rm {your_rm_config_entry} --rng_seed {your_choice_seed}
```

## Best-of-*n* (BoN) inference
Best-of-*n* inference (also known as re-ranking) can be performed using a base policy model (post-SFT) and one or more trained reward models, as follows:
```
python src/bon/run_bon_pipeline.py {your_reward_models_path}
```
This command is customizable with the following arguments and options:
| Argument        | Type | Required |  Description                                              | Default value              |      
| --------------- | ---- | -------- | --------------------------------------------------------- | -------------------------- |
| `proxy_rm_path`             | str         | yes      | generic path to proxy (non-gold) reward models to use. This should be a string with a "{seed}" placeholder, so that multiple reward models can be retrieved, both for ensembles and general convenience. e.g. "models/rm-pythia-44m_seed{seed}"| - |                                                                           
| `output_dir`                | str         | no       | name of (path to) the directory where the output model should be saved. This will go under [runs/](/runs/). | bon_sampling_{curr_time}                                              
| `gold_gens` | str         | no       | name of (path to) BoN dataset containing at least `big_n` answers (see [dataset guide](/src/data_utils/README.md#best-of-n-bon) for details)| tlc4418/gold_labelled_gens |                                     
| `big_n`                     | int         | no       | total number of answers to perform BoN sampling over. 'N' in the unbiased estimator formula. Usually the total number of answers in the dataset, will be used to cut down the dataset otherwise. | 12600 |                                                                       
| `sample_ns`                 | str         | no       | list of indexes (the *n* in best-of-*n*) at which to perform BoN sampling. Comma-separated list of ints. | "1,2,4,8,16,32,64,128,<br>256,512,1024,2048,<br>4096,6144,8192,12500" |
| `seeds`                     | str         | no       | list of seeds corresponding to which reward models to run (will be used to fill the "{seed}" placeholder for the `proxy_rm_path`). If doing BoN for ensembles, the length of this list is also your ensemble cardinality. Comma-separated list of ints. | "1,2,3,4,5"  |                                                                
| `ensembles`                 | bool        | no       | whether to run BoN over ensembles. If set to true, BoN for the three types of ensembles in the paper (mean, WCO, UWO) will be performed in addition to the individual reward models. | True  |                                                                       
| `uwo_weights`               | str         | no       | list of UWO weights to use when doing BoN sampling with the UWO ensemble (if `ensembles` is true). Results will be given for a new ensemble with each weight. Comma-separated list of floats. | "0.5" |                                                                       


A help function is also provided (`--help`) which will display a condensed version of the above, including the appropriate flag names for specifying each parameter.

Our implementation uses an unbiased estimator (see paper for details) for robust and unbiased results.

The relevant result of running this command will be a "bon_sampled_results.json" results file for each run (one for each individual seed and one for each ensemble if desired). This file contains a list of dictionary entries, where each entry contains the sampled index *n*, the proxy reward model score at that *n*, and the corresponding gold reward model score at that *n*, as follows:
```
[
    {
        "n": {some_int},
        "proxy_score": {proxy_score},
        "gold_score": {gold_score}
    },

    ...
]
```
These data points can then be used to plot the BoN performance of different policies according to both proxy and gold reward model score, as a function of *n*.


## RL training (PPO)
RL training with PPO can be performed using a base policy model (post-SFT) and one or more trained reward models. The policy will be trained using PPO and the reward models as a reward function. We provide options for both single reward models and ensembles, along with conservative optimization techniques. Example configs entries for this are provided in the [RL config](/configs/config_rl.yaml). We detail some of the following important fields (which will likely be changed the most) for a new config entry:

| Field name    | Subfield name      | Type  |  Description                                                              | Example value |
| ------------- | ------------------ | ----- | ----------------------------------------------------------------------- | ------------- |
| `output_dir`  | -                  | str   | name of (path to) the directory where the outputs should be saved. | runs/ppo |
| `datasets`    | -                  | list  | list of instruction datasets to use for training (see [dataset guide](/src/data_utils/README.md) for details) | - alpaca_farm |
| `gold_config` | `model_name`       | str   | name of (path to) the 'gold' reward model to use to evaluate policy outputs alongside the proxy reward mdoel rewards | alpaca_farm_models/<br>reward-model-human
|               | `is_alpacafarm_rm` | bool  | whether or not the gold reward model is from AlpacaFarm. Because the input format is different for their models, different dataset processing is needed for evaluation. | true
| `rank_config` | `model_names`      | str   | list of names/paths for the proxy reward models. A single path should be provided for single reward model PPO training, and several paths for ensembles. When providing a single path, it can also contain a '{seed}' placeholder, which can be replaced by passing an additional `--rm_seed {your_seed}` argument to the run command given below. This can be helpful when wanting to run multiple seeds in parallel using job managers, without requiing a different config each time. | - models/rm-pythia-44m_seed1
|               | `objective_name`   | str   | name of the conservative optimization objective to use to combine rewards from the reward model ensemble members. Must be one of "mean", "WCO", or "UWO". Only used if more than one model is provided in `model_names` | mean
|               | `uwo_weight`       | float | weight (&lambda; coefficient) for UWO conservative optimization objective. Only used when `objective_name` is "UWO". | 0.1
| `sft_config`  | `model_name`       | str   | name of (path to) the policy model (post-SFT) to use for PPO training. This is the model that will be trained. | tlc4418/pythia_1.4b_sft_policy


Many default hyperparameters can also be overwritten, both in the [RL config](/configs/config_rl.yaml) and more specific PPO ones in the [PPO config file](/configs/config_ppo.yaml)


PPO training (and subsequent gold reward model evaluation) can be started with the following command:
```
accelerate launch --config_file configs/accelerate_config.yaml src/ppo/trainer_rl.py --configs defaults defaults_rlhf {your_rm_config_entry}
```

Gold reward model evaluation is performed after training is completed so as to avoid loading the large gold reward mdoel during training. KL divergence from the initial policy is also recorded at each evaluation step, such that the evaluation results in the "eval/" output folder can be used to plot both proxy and gold reward model score either as a function of steps or of KL divergence. Please see the [AlpacaFarm respository](https://github.com/tatsu-lab/alpaca_farm) for instructions on downloading the 7B reward model used as the gold reward model in our paper (`reward-model-human`)


## Example usage (full pipeline) - single reward model

### SFT models
Simply use the provided* [tlc4418/pythia_1.4b_sft_policy](https://huggingface.co/tlc4418/pythia_1.4b_sft_policy) and [tlc4418/pythia_70m_sft](https://huggingface.co/tlc4418/pythia_70m_sft) models.

Alternatively, you could run your own SFT training for the policy model and base reward model, following the [SFT instructions](#supervised-fine-tuning-sft) above.

### Reward models
Train a reward model from the base reward model (post-SFT), with any rng seed (here "1"):
```
accelerate launch --config_file configs/accelerate_config.yaml src/reward_modeling/training/trainer_rm.py --configs defaults_rm rm-pythia-44m --rng_seed 1
```

### BoN sampling
Run BoN sampling for the trained reward model:
```
python src/bon/run_bon_pipeline.py models/rm-pythia-44m_seed{seed} --seeds 1
```

### PPO RL training
For PPO, run the following command:
```
accelerate launch --config_file configs/accelerate_config.yaml src/ppo/trainer_rl.py --configs defaults defaults_rlhf pythia_rlhf_individual
```

Here we set the config to already contain the trained reward model under the rank_config's `model_names`, but you will want to change this to match yours when customizing this process.


## Example usage (full pipeline) - ensemble of *k* reward models

### SFT models
Simply use the provided [tlc4418/pythia_1.4b_sft_policy](https://huggingface.co/tlc4418/pythia_1.4b_sft_policy) and [tlc4418/pythia_70m_sft](https://huggingface.co/tlc4418/pythia_70m_sft) models.

Alternatively, you could run your own SFT training for the policy model and base reward model, following the [SFT instructions](#supervised-fine-tuning-sft) above.

### Reward models
Train as many (*k*) reward models as you want from the base reward model (post-SFT). Just run the following *k* times, changing the seed each time (e.g. 1-5):
```
accelerate launch --config_file configs/accelerate_config.yaml src/reward_modeling/training/trainer_rm.py --configs defaults_rm rm-pythia-44m --rng_seed {seed}
```

### BoN sampling
Assuming you have trained 5 reward models above, this will run BoN sampling for all 5 models and the 3 ensembles types (mean, WCO, UWO):
```
python src/bon/run_bon_pipeline.py models/rm-pythia-44m_seed{seed} --seeds 1,2,3,4,5 --ensembles
```

### PPO RL training
For PPO, you will want to run the following command for each ensemble you want to train, making sure the appropriate conservative optimization `objective` and `uwo_weight` (if applicable) are set in the [RL config]((/configs/config_rl.yaml)):
```
accelerate launch --config_file configs/accelerate_config.yaml src/ppo/trainer_rl.py --configs defaults defaults_rlhf pythia_rlhf_ensemble
```

Here we set the config to already contain the 5 trained reward models under the rank_config's `model_names`, but you will want to change these to match yours when customizing this process.




## Citation
If you use the models, data, or code in this repo, please consider citing our work.
```
@article{coste2023reward,
  title={Reward model ensembles help mitigate overoptimization},
  author={Coste, Thomas and Anwar, Usman and Kirk, Robert and Krueger, David},
  journal={arXiv preprint arXiv:2310.02743},
  year={2023}
}
```