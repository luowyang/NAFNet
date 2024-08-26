from __future__ import annotations

from typing import Any, Callable, Iterator, Mapping

import torch
from torch.utils.data import DataLoader


def iterative_map(func: Callable[[torch.Tensor], Any], x):
    r"""Iteratively map x with func.
    x may be Tensor, dict or list of Tensor, or dict or list of iterative structure.
    func takes a Tensor or ndarray, return anything or nothing.

    Returns:
        The returned values of func with the same structure as x
    """
    if isinstance(x, torch.Tensor):
        out = func(x)
    elif isinstance(x, tuple) and hasattr(x, '_fields'):  # namedtuple
        out = type(x)(*(iterative_map(func, samples) for samples in x))
    elif isinstance(x, (list, tuple)):  # `str` will miserably fail if we use `typing.Sequence`
        out = type(x)((iterative_map(func, samples) for samples in x))
    elif isinstance(x, Mapping):
        out = type(x)({k: iterative_map(func, v) for k, v in x.items()})
    else:
        out = x  # do nothing for other types
    return out


class DataPrefetcher:
    def __init__(self, dataloader: DataLoader, device=None, dtype=None):
        if not dataloader.pin_memory:
            raise ValueError(f'{self.__class__.__name__} requires dataloader `pin_memory=True`')
        self.dataloader = dataloader
        self.device = device
        self.dtype = dtype
        self.stream = torch.cuda.Stream()

    def _preload(self, loader_iter: Iterator):
        try:
            batch = next(loader_iter)
        except StopIteration:
            batch = None
        with torch.cuda.stream(self.stream):
            batch = iterative_map(lambda x: x.to(device=self.device, dtype=self.dtype, non_blocking=True), batch)
        return batch

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        loader_iter = iter(self.dataloader)
        next_batch = self._preload(loader_iter)
        while True:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            next_batch = self._preload(loader_iter)
            yield batch

    @property
    def sampler(self):
        return self.dataloader.sampler
