#!/usr/bin/env python
# coding=utf-8
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .bases import GIRDataset


def balance_datasets_by_dataset(datasets: Iterable[GIRDataset]) -> list[GIRDataset]:
    r"""Balances `datasets` by dataset."""
    balanced_datasets = []
    max_len = len(max(datasets, key=len))
    for dataset in datasets:
        len_ratio = int(round(max_len / len(dataset)))
        balanced_datasets.extend([dataset] * len_ratio)
    return balanced_datasets


def balance_datasets_by_task(datasets: Iterable[GIRDataset]) -> list[GIRDataset]:
    r"""Balances `datasets` by task."""
    balanced_datasets = []

    # Explain: first scale up datasets per task uniformly, then scales up the least dataset in each task.
    datasets_by_task = defaultdict(list)
    for dataset in datasets:
        datasets_by_task[dataset.task].append(dataset)
    datasets_by_task = list(datasets_by_task.values())
    lens_by_task = [sum(map(len, x)) for x in datasets_by_task]
    max_len = max(lens_by_task)

    for datasets_per_task, len_per_task in zip(datasets_by_task, lens_by_task):
        ratios_per_task = [max_len // len_per_task] * len(datasets_per_task)
        left_length = max_len - ratios_per_task[0] * len_per_task

        while left_length > 0:
            # find min_len after scaling
            arg_min_len, min_len = 0, (ratios_per_task[0] * len(datasets_per_task[0]))
            for index, dataset in enumerate(datasets_per_task):
                scaled_len = ratios_per_task[index] * len(dataset)
                if scaled_len < min_len:
                    arg_min_len, min_len = index, scaled_len

            # scale up by one if possible
            if len(datasets_per_task[arg_min_len]) <= left_length:
                ratios_per_task[arg_min_len] += 1
                left_length -= len(datasets_per_task[arg_min_len])
            else:
                # this prevents over-scaling of small datasets at the cost of greater imbalance
                break

        for ratio, dataset in zip(ratios_per_task, datasets_per_task):
            balanced_datasets.extend([dataset] * ratio)

    return balanced_datasets


def balance_datasets_no(datasets: Iterable[GIRDataset]) -> list[GIRDataset]:
    r"""Dummy, does nothing."""
    return list(datasets)


DATASET_BALANCERS = {
    'by_dataset': balance_datasets_by_dataset,
    'by_task': balance_datasets_by_task,
    'no': balance_datasets_no,
}


def balance_datasets(datasets: Iterable[GIRDataset], strategy: str = 'by_dataset') -> list[GIRDataset]:
    r"""Balances `datasets` according to `strategy` to mitigate the long-tailed distribution problem.

    Args:
        datasets: Datasets to balance.
        strategy: Balancing stategy, must be one of "no", "by_dataset" (default) and "by_task".

    Returns:
        List of balanced datasets.
    """
    balancer = DATASET_BALANCERS.get(strategy, None)

    if balancer is None:
        raise ValueError(f'unknown strategy "{strategy}"; expects one of {list(DATASET_BALANCERS)}')

    return balancer(datasets)
