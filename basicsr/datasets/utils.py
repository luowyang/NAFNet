from __future__ import annotations

from PIL import Image
import random
import warnings
from typing import Sequence


def pil_load_rgb(path: str) -> Image.Image:
    r"""Load PIL Image in HWC RGB format from `path`."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def random_split(*columns: Sequence[Sequence], rate: float = 0.9, split: str = 'train', seed: int = 2023):
    r"""Randomly splits data cloumns into two splits, and returns train or val
    split in columnar format."""
    assert 0 <= rate <= 1, f'expects rate to be [0, 1]; gets {rate}'

    max_len = len(max(columns, key=len))
    # transpose columns into rows
    rows = list(zip(*columns))
    if len(rows) < max_len:
        warnings.warn(f'some data are missing, truncate {max_len} to {len(rows)}')

    rng = random.Random(seed)
    rng.shuffle(rows)

    split_index = int(len(rows) * rate)
    split_rows = rows[:split_index] if split == 'train' else rows[split_index:]

    # transpose rows back to columns
    split_columns = list(zip(*split_rows))
    if len(columns) == 1:
        split_columns = split_columns[0]

    return split_columns
