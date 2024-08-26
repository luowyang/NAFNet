from __future__ import annotations

import enum
import json
import os
from typing import Sequence, Any, Callable, Optional

import numpy as np
import torch
import torch.utils.data
from PIL import Image

from .prompt import get_prompt
from .utils import pil_load_rgb


# HACK: allow pytorch's default collate_fn to process PIL Image
from torch.utils.data._utils.collate import collate_str_fn, default_collate_fn_map

default_collate_fn_map.setdefault(Image.Image, collate_str_fn)


@enum.unique
class DataType(enum.IntEnum):
    r"""Type of the dataset."""
    TRAIN_TEST = 0  # the dataset contains train and test splits
    TRAIN_ONLY = 1  # the dataset only contains train split
    TEST_ONLY  = 2  # the dataset only contains test split


class GIRDataset(torch.utils.data.Dataset):
    r"""Base class of all general image restoration (GIR) datasets.

    A GIRDataset belongs to one (and only one) task, and may optionally contain several subsets.
    """

    def __init__(
        self,
        annotations: str | Sequence[dict[str, str]],
        split: str,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cached_latent_root: Optional[str] = None,  # for VAE training
    ) -> None:
        self.transform = transform
        self.cached_latent_root = cached_latent_root

        if isinstance(annotations, str):
            if os.path.isfile(annotations):
                with open(annotations, 'r') as f:
                    annotations = json.load(f)
            elif os.path.isdir(annotations):
                annotations = self.make_annotations(annotations, split)
            else:
                raise ValueError(f'unknown `annotations={annotations}`')
        
        if not isinstance(annotations, Sequence):
            raise TypeError(f'expects `annotations` to be a sequence; gets {type(annotations)}')
        if not all(isinstance(anno, dict) for anno in annotations):
            raise TypeError('`annotations` must be a sequence of dict')
        
        self.annotations: list[dict[str, str]] = annotations

        self._task = self._get_task(self.annotations[0]['lq'])
    
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        """Make annotations from `root` and `split`."""
        raise NotImplementedError

    def load_lq_image(self, path: str) -> Image.Image:
        """Loads HWC format RGB LQ image from `path`.
        Overrides this method if you have custom loading logics."""
        return pil_load_rgb(path)

    def load_hq_image(self, path: str) -> Image.Image:
        """Loads HWC format RGB HQ image from `path`.
        Overrides this method if you have custom loading logics."""
        return pil_load_rgb(path)

    def _get_task(self, path: str) -> str:
        return path.split('All_in_one')[1].split(os.sep)[1]

    @property
    def task(self) -> str:
        return self._task

    def __getitem__(self, index: int):
        annotation = self.annotations[index]

        lq_path, hq_path = annotation['lq'], annotation['hq']
        lq_image, hq_image = self.load_lq_image(lq_path), self.load_hq_image(hq_path)
        # NOTE: currently super resolution is not supported.
        assert lq_image.size == hq_image.size, (lq_image.size, hq_image.size)

        # format: (width, height), same as PIL Image
        original_size = torch.LongTensor([lq_image.size[0], lq_image.size[1]])

        data = dict(lq_image=lq_image, hq_image=hq_image)
        if self.transform is not None:
            data = self.transform(data)

        # for debug
        task = self._get_task(lq_path)
        assert self.task == task, (self.task, task)

        prompt_np = get_prompt(self.task)
        prompt_image = Image.fromarray(prompt_np, mode='RGB')
        prompt_np = prompt_np.astype(np.float32) / 255  # pixel value range: [0, 1]
        prompt_tensor = torch.from_numpy(prompt_np.transpose(2, 0, 1))

        data.update(
            task=self.task,
            dataset=self.__class__.__name__,
            subset=annotation.get('subset', ''),
            original_size=original_size,
            prompt_image=prompt_image,
            prompt_tensor=prompt_tensor,
            lq_path=lq_path,
            hq_path=hq_path,
        )

        # Load cached latents, only VAE training requires it.
        if self.cached_latent_root is not None:
            lq_stem = os.path.splitext(os.path.basename(data['lq_path']))[0]
            saved_path = os.path.join(
                self.cached_latent_root, data['task'], data['dataset'], data['subset'], lq_stem + '.pt'
            )
            latent = torch.load(saved_path, map_location='cpu')
            data.update(latent=latent)

        return data

    def __len__(self) -> int:
        return len(self.annotations)
