# RL training (PPO)
An explanation for [running PPO](#running-ppo) is provided in the project-level README, but we also provide it here, with some additional sections, for convenience.


## Running PPO
RL training with PPO can be performed using a base policy model (post-SFT) and one or more trained reward models. The policy will be trained using PPO and the reward models as a reward function. We provide options for both single reward models and ensembles, along with conservative optimization techniques. Example configs entries for this are provided in the [RL config file](/configs/config_rl.yaml). We detail some of the following important fields (which will likely be changed the most) for a new config entry:

| Field name    | Subfield name      | Type  | Description                                                             | Example value |
| ------------- | ------------------ | ----- | ----------------------------------------------------------------------- | ------------- |
| `output_dir`  | -                  | str   | name of (path to) the directory where the outputs should be saved. | runs/ppo |
| `datasets`    | -                  | list  | list of instruction datasets to use for training (see [dataset guide](/src/data_utils/README.md) for details) | - alpaca_farm |
| `gold_config` | `model_name`       | str   | name of (path to) the 'gold' reward model to use to evaluate policy outputs alongside the proxy reward mdoel rewards | alpaca_farm_models/<br>reward-model-human
|               | `is_alpacafarm_rm` | bool  | whether or not the gold reward model is from AlpacaFarm. Because the input format is different for their models, different dataset processing is needed for evaluation. | true
| `rank_config` | `model_names`      | str   | list of names/paths for the proxy reward models. A single path should be provided for single reward model PPO training, and several paths for ensembles. When providing a single path, it can also contain a '{seed}' placeholder, which can be replaced by passing an additional `--rm_seed {your_seed}` argument to the run command given below. This can be helpful when wanting to run multiple seeds in parallel using job managers, without requiing a different config each time. | - models/rm-pythia-44m_seed1
|               | `objective_name`   | str   | name of the conservative optimization objective to use to combine rewards from the reward model ensemble members. Must be one of "mean", "WCO", or "UWO". Only used if more than one model is provided in `model_names` | mean
|               | `uwo_weight`       | float | weight (&lambda; coefficient) for UWO conservative optimization objective. Only used when `objective_name` is "UWO". | 0.1
| `sft_config`  | `model_name`       | str   | name of (path to) the policy model (post-SFT) to use for PPO training. This is the model that will be trained. | tlc4418/pythia_1.4b_sft_policy


Many default hyperparameters can also be overwritten, both in the [RL config file](/configs/config_rl.yaml) and more specific PPO ones in the [PPO config file](/configs/config_ppo.yaml)


PPO training (and subsequent gold reward model evaluation) can be started with the following command:
```
accelerate launch --config_file configs/accelerate_config.yaml src/ppo/trainer_rl.py --configs defaults defaults_rlhf {your_rm_config_entry}
```

Gold reward model evaluation is performed after training is completed so as to avoid loading the large gold reward mdoel during training. KL divergence from the initial policy is also recorded at each evaluation step, such that the evaluation results in the "eval/" output folder can be used to plot both proxy and gold reward model score either as a fucntion of steps or of KL divergence.

## Running only gold reward model evaluation
If you have already trained a policy model and have evaluation logs with KL divergence and proxy scores but no gold scores, you ay want to run only the gold score evaluation. This could happen if, for example, you were to cancel training midway but had already collected several proxy evaluation steps and wanted to look at their gold scores. You can run the following script for this:
```
python src/ppo/run_ppo_gold_eval.py {your_current_eval_dir} {your_gold_rm_path}
```

The arguments this script takes are as follows (you can also see a condensed version by runnining the scipt with "--help"):
| Argument              | Type        | Required | Description                                      | Default value  |
| --------------------- | ----------- | -------- | ------------------------------------------------ | -------------- |
| `eval_dir`                  | str         | yes      | path to the folder containing the current evaluation files containing proxy scores for every evaluation step. Gold scores will be saved to the same files. | -                                                                            | 
| `gold_rm_model_name`        | str         | yes       | name of (path to) the 'gold' reward model to use for producing gold evaluation scores | -                                                                           | 
| `is_alpacafarm_rm`          | bool        | no       | whether or not the gold reward model is from AlpacaFarm. Because the input format is different for their models, different dataset processing is needed for evaluation. | False                                                                        | 
| `batch_size`                | int         | no       | batch size used when prompting the gold RM for evaluations |32                                                                           | 