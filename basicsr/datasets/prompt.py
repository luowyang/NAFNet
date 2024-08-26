from __future__ import annotations

import copy
from functools import partial
import random

import numpy as np
import torch
import torch.utils.data

from basicsr.transforms import prompt_aug

from .utils import pil_load_rgb

__all__ = [
    'PROMPT_STRATEGIES',
    'get_train_collate_fn',
]


PROMPT_PATH = 'assets/prompt.jpeg'
PROMPT_IMG = pil_load_rgb(PROMPT_PATH).resize((224, 224))
PROMPT_NUMPY = np.asarray(PROMPT_IMG)  # HWC, uint8
PROMPT_TENSOR = torch.from_numpy(PROMPT_NUMPY.transpose(2, 0, 1).astype(np.float32) / 255)  # CHW, float32

PROMPT_AUGS = {
    'Image_Deraining': prompt_aug.rainline_related,
    'Raindrop_Removal': prompt_aug.rainline_related,
    'Image_Desnowing': prompt_aug.snow_related,
    'Image_Dehazing': prompt_aug.fog_related,
    'Defocus_deblurring': prompt_aug.defocus_related,
    'Gaussian_denoising': prompt_aug.noise_related,
    'Low_light_enhancement': prompt_aug.low_light_related,
    'Motion_deblurring': prompt_aug.motion_related,
    'Real_denoising': prompt_aug.noise_related,
}


def get_prompt(task: str) -> np.ndarray:
    r"""Gets a prompt image (with distortions according to `task`) as NumPy array of uint8, HWC format, RGB."""
    assert task in PROMPT_AUGS, f'{task} is not in {set(PROMPT_AUGS.keys())}'

    # TODO: verify that all augments accept RGB images.
    if task in ['Defocus_deblurring', 'Low_light_enhancement']:
        # NOTE: these augments are implemented by `albumentations` which accepts a single image as input.
        prompt = PROMPT_AUGS.get(task, prompt_aug.noise_related)(image=PROMPT_NUMPY)["image"]
    else:
        prompt = PROMPT_AUGS.get(task, prompt_aug.noise_related)(image=PROMPT_NUMPY)

    prompt = np.asarray(prompt)
    assert prompt.dtype == np.uint8, (prompt.shape, prompt.dtype)

    return prompt


def add_empty_collate_fn(batch: list[dict]) -> dict:
    batch2 = copy.deepcopy(batch)
    for data in batch:
        data['task'] = ''
        data['hq_path'] = data['lq_path']
        data['hq_image'] = data['lq_image']
        data['hq_tensor'] = data['lq_tensor']
    batch.extend(batch2)
    random.shuffle(batch)
    return torch.utils.data.default_collate(batch)


def shuffle_collate_fn(batch: list[dict]) -> dict:
    # Guard against dumbass that forgets to shuffle the train data.
    random.shuffle(batch)
    # Shuffle tasks of the second half.
    first_half, second_half = batch[:round(len(batch)/2)], batch[round(len(batch)/2):]
    shuffled_tasks = [data['task'] for data in second_half]
    random.shuffle(shuffled_tasks)
    for data, new_task in zip(second_half, shuffled_tasks):
        if data['task'] != new_task:
            data['task'] = new_task
            data['hq_path'] = data['lq_path']
            data['hq_image'] = data['lq_image']
            data['hq_tensor'] = data['lq_tensor']
    # Extend the second half into the first half.
    first_half.extend(second_half)
    # Final shuffle and collate.
    random.shuffle(first_half)
    return torch.utils.data.default_collate(first_half)


def cfg_collate_fn(batch: list[dict], cfg_drop_prob: float = 0.1) -> dict:
    # Randomly sample the indices to drop prompts.
    drop_ids = np.arange(len(batch))[np.random.random_sample(len(batch)) < cfg_drop_prob].tolist()
    # Drop prompts by substituting them with empties.
    for index in drop_ids:
        batch[index]['task']            = ''
        batch[index]['prompt_image']    = PROMPT_IMG.copy()
        batch[index]['prompt_tensor']   = PROMPT_TENSOR.clone()
    return torch.utils.data.default_collate(batch)


PROMPT_STRATEGIES = {
    'add_empty': add_empty_collate_fn,
    'shuffle': shuffle_collate_fn,
    'no': torch.utils.data.default_collate,
}


def get_train_collate_fn(prompt_strategy: str = 'no', cfg_drop_prob: float = 0):
    r"""Gets a collate function for training.

    Args:
        prompt_strategy (str):
            One of the following strategies:
            - `add_empty`: the returned collate function repeats the batch
              but replaces prompts with emprty prompts, and replaces HQs by LQs.
            - `shuffle`: shuffle half of the prompts, and replaces HQs by LQs
              if the prompts mismatch after shuffling.
            - `no` (default): normal.
    """
    if cfg_drop_prob > 0:
        assert prompt_strategy == 'no', f'`{prompt_strategy=}` must be `no` when `{cfg_drop_prob=}` > 0'
        return partial(cfg_collate_fn, cfg_drop_prob=cfg_drop_prob)
    
    collate_fn = PROMPT_STRATEGIES.get(prompt_strategy, None)

    if collate_fn is None:
        raise ValueError(f'unknown strategy "{prompt_strategy}"; expects one of {list(PROMPT_STRATEGIES)}')

    return collate_fn
