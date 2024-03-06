import os
import json
import typer
import time
from typing_extensions import Annotated
from datasets import load_dataset
from src.bon.bon_sampling import bon_sample
from src.bon.ensemble_rm import get_ensemble_score
from src.bon.utils import prep_files

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_ensembles(
    output_dir: Annotated[
        str,
        typer.Argument(
            help="The name of the experiment output folder, under /runs. Should "
            "already contain BoN results for the individual RM seeds provided."
        ),
    ] = f"bon_sampling_run_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}",
    sample_ns: Annotated[
        list[int],
        typer.Option("--sample-ns", "-ns", help="What indexes (n) to perform BoN sampling at."),
    ] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12500],
    seeds: Annotated[list[int], typer.Option(help="Which individual RM seeds to run.")] = [1, 2, 3],
    uwo_weights: Annotated[list[float], typer.Option(help="The weights to use for UWO ensemble.")] = [0.5],
):
    """
    Run BoN, only for ensembles. This assumes you have already run the individual RMs
    and that their results are stored in the provided output directory, under /runs.

    Big N will match the value used for the individual RMs.
    """

    # Ensembles BoN inference, if required, using the individual RM results ------------
    types = ["mean", "WCO"] + ["UWO"] * len(uwo_weights)
    for i, ensemble_type in enumerate(types):
        print(f"Running ensemble {ensemble_type}...")

        # Output files
        weight = uwo_weights[i - 2] if ensemble_type == "UWO" else None
        weight_str = f"-{weight}" if weight else ""
        parent = f"runs/{output_dir}/ensemble_{ensemble_type}{weight_str}"
        f"_seeds{'-'.join(map(str, seeds))}"
        proxy_labelled_answers_filename, bon_samples_filename = prep_files(parent)

        ### Get ensemble scores from individual RMs --------------------------------
        print("Getting ensemble proxy scores...")

        proxy_scores_format = f"runs/{output_dir}/seed{'{seed}'}/proxy_labelled_sorted.json"
        datasets = [load_dataset("json", data_files=proxy_scores_format.format(seed=s))["train"] for s in seeds]
        ensemble_data = get_ensemble_score(datasets, ensemble_type, weight)

        with open(proxy_labelled_answers_filename, "w", encoding="utf-8") as f:
            json.dump(ensemble_data, f, ensure_ascii=False, indent=4)

        ### Perform best-of-n sampling ---------------------------------------------
        print("Performing BoN sampling...")

        scores = bon_sample(ensemble_data, sample_ns)

        with open(bon_samples_filename, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)

        print(f"Ensemble {ensemble_type} done.")

    print("BoN ensembles complete.")


if __name__ == "__main__":
    typer.run(run_ensembles)
