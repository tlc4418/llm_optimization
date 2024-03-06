import numpy as np


def mean_objective_function(scores):
    return np.mean(scores)


def worst_case_optimization(scores):
    return np.min(scores)


def uncertainty_weighted_optimization(scores, coeff=1):
    return np.mean(scores) - coeff * np.var(scores)


OBJECTIVE_FUNCTIONS = {
    "mean": mean_objective_function,
    "WCO": worst_case_optimization,
    "UWO": uncertainty_weighted_optimization,
}


def get_ensemble_score(datasets, objective_function_name="mean", weight=None):
    objective_function = OBJECTIVE_FUNCTIONS[objective_function_name]

    ensemble_data = []
    for i in range(len(datasets[0])):
        entries = [dataset[i] for dataset in datasets]
        ensemble_entry = entries[0]

        ensemble_proxy_scores = []
        for answer in entries[0]["answers"]:
            # get proxy score of answer for each dataset, and normalize
            proxy_scores = [
                (entry["proxy_scores"][entry["answers"].index(answer)])  # offsets[j][0])
                # / math.sqrt(offsets[j][1])
                for j, entry in enumerate(entries)
            ]

            # get score for ensemble
            ensemble_score = objective_function(proxy_scores, weight) if weight else objective_function(proxy_scores)
            ensemble_proxy_scores.append(ensemble_score)

        indices = np.argsort(ensemble_proxy_scores)
        ensemble_entry["proxy_scores"] = [ensemble_proxy_scores[i] for i in indices]
        ensemble_entry["answers"] = [ensemble_entry["answers"][i] for i in indices]
        # ensemble_entry["gold_scores"] = [
        #     (ensemble_entry["gold_scores"][i])#offsets[0][2])
        #     #/ math.sqrt(offsets[0][3])
        #     for i in indices
        # ]

        ensemble_data.append(ensemble_entry)

    return ensemble_data
