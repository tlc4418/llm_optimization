from pathlib import Path


def prep_files(parent: str):
    """Prepare the output files."""

    Path(parent).mkdir(parents=True, exist_ok=True)
    proxy_labelled_answers_filename = f"{parent}/proxy_labelled_sorted.json"
    bon_samples_filename = f"{parent}/bon_sampled_results.json"

    return proxy_labelled_answers_filename, bon_samples_filename
