import scipy
import numpy as np
from datasets import load_dataset, Dataset
from typing import Union
from multiprocessing import Pool
import functools
from tqdm import tqdm

POOL_SIZE = 10


def unbiased_sampler(dataset, n: int, N: int):
    coeffs = np.array(
        [scipy.special.comb(i - 1, n - 1, exact=True) / scipy.special.comb(N, n, exact=True) for i in range(n, N + 1)]
    )

    proxy_scores = []
    gold_scores = []
    for question in dataset:
        proxy_scores.append(question["proxy_scores"][n - 1 : N])
        gold_scores.append(question["gold_scores"][n - 1 : N])

    # Get proxy scores
    proxy_total = np.mean(np.sum(coeffs * proxy_scores, axis=1))

    # Get gold RM scores
    gold_total = np.mean(np.sum(coeffs * gold_scores, axis=1))

    return gold_total, proxy_total


def get_bon_entry(n, dataset, big_n: int):
    gold_score, proxy_score = unbiased_sampler(dataset, n, big_n)
    return {
        "n": n,
        "proxy_score": proxy_score,
        "gold_score": gold_score,
    }


def bon_sample(
    dataset: Union[str, list],
    ns: list,
):
    # Dataset labelled and sorted by proxy RM score
    dataset = (
        load_dataset("json", data_files=dataset)["train"] if isinstance(dataset, str) else Dataset.from_list(dataset)
    )

    big_n = len(dataset[0]["proxy_scores"])
    ns = [n for n in ns if n <= big_n]
    with Pool(POOL_SIZE) as p:
        bon_scores = p.map(
            functools.partial(
                get_bon_entry,
                dataset=dataset,
                big_n=big_n,
            ),
            tqdm(ns),
        )

    return list(bon_scores)
