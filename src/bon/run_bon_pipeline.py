import os
import json
import typer
import time
from typing_extensions import Annotated
from datasets import load_dataset
from src.reward_modeling.scoring.score import score_answers
from src.bon.bon_sampling import bon_sample
from src.bon.run_bon_ensembles import run_ensembles
from src.bon.utils import prep_files

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    proxy_rm_path: Annotated[
        str,
        typer.Argument(
            help="The path to the proxy RM model to use. This should be a string with " "a {seed} placeholder."
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option("-o", help="The name of the experiment output folder, to be put in /runs."),
    ] = f"bon_sampling_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}",
    gold_labelled_generations: Annotated[
        str,
        typer.Option(
            "--gold-labelled-generations",
            "-gold",
            help="Dataset with gold-labelled generations.",
        ),
    ] = "tlc4418/gold_labelled_gens",
    big_n: Annotated[
        int,
        typer.Option(
            help="Total number of answers to perform BoN sampling over. 'N' in the "
            "unbiased estimator formula. Usually the total number of answers in the "
            "dataset."
        ),
    ] = 12600,
    sample_ns: Annotated[
        str,
        typer.Option(
            "--sample-ns",
            "-ns",
            help="Comma-separated list of ints. What indexes (n) " "to perform BoN sampling at.",
        ),
    ] = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,6144,8192,12500",
    seeds: Annotated[
        str,
        typer.Option(help="Comma-separated list of ints. Which individual RM " "seeds to run."),
    ] = "1,2,3,4,5",
    ensembles: Annotated[bool, typer.Option(help="Whether to run ensembles.")] = False,
    uwo_weights: Annotated[
        str,
        typer.Option(help="Comma-separated list of floats. The weights to use for " "UWO ensemble."),
    ] = "0.5",
):
    """Run BoN."""

    print("Starting BoN pipeline...")

    # Process parameter string lists
    sample_ns = [int(n) for n in sample_ns.split(",")]
    seeds = [int(s) for s in seeds.split(",")]
    uwo_weights = [float(w) for w in uwo_weights.split(",")]

    # Reduce the number of answers to big_n if needed
    gold_labelled_generations = load_dataset(gold_labelled_generations)["validation"]

    def _truncate_answers(entry):
        entry["answers"] = entry["answers"][:big_n]
        entry["gold_scores"] = entry["gold_scores"][:big_n]
        return entry

    gold_labelled_generations = gold_labelled_generations.map(_truncate_answers, batched=True, batch_size=10)

    for s in seeds:
        print(f"Running seed {s}...")
        # Models
        proxy_rm_model_name = proxy_rm_path.format(seed=s)

        # Where to store the proxy RM labelled and sorted answers, and the BoN samples
        parent = f"runs/{output_dir}/seed{s}"
        proxy_labelled_answers_filename, bon_samples_filename = prep_files(parent)

        # Label the dataset with the proxy RM score for each answer, and sort ------------------------------------------
        print("Scoring answers...")

        data = score_answers(
            model_name=proxy_rm_model_name,
            dataset=gold_labelled_generations,
            scores_type="proxy_scores",
            sort=True,
        )

        with open(proxy_labelled_answers_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # Perform best-of-n sampling -----------------------------------------------------------------------------------
        print("Performing BoN sampling...")

        scores = bon_sample(data, sample_ns)

        with open(bon_samples_filename, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)

        print(f"Seed {s} done.")

    # Ensembles BoN inference, if required, using the individual RM results --------------------------------------------
    if ensembles:
        run_ensembles(
            output_dir=output_dir,
            sample_ns=sample_ns,
            seeds=seeds,
            uwo_weights=uwo_weights,
        )

    print("BoN pipeline complete.")


if __name__ == "__main__":
    typer.run(main)
