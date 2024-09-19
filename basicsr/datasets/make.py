from __future__ import annotations

import math

import cv2
import numpy as np
import os
import torch
import torch.utils.data
from torchvision import transforms as tv_trans
from torchvision.transforms import functional as tv_f
from PIL import Image
from typing import Optional, TypeVar

from basicsr.datasets.bases import GIRDataset
from basicsr.datasets.datasets import DATASETS
from basicsr.logger import get_logger
from basicsr.utils.windows_crop import merge_crops, sliding_window_crop

logger = get_logger(__name__)

T = TypeVar('T')


def to_2tuple(value):
    if not isinstance(value, tuple):
        return (value, value)
    assert len(value) == 2, value
    return value


def image_to_numpy_unnormalized(img: Image.Image) -> np.ndarray:
    r"""Converts a PIL RGB Image to NumPy array of range `[0, 1]` in HWC format."""
    assert isinstance(img, Image.Image), type(img)
    assert img.mode == 'RGB', img.mode
    img_np = np.asarray(img, dtype=np.float32) / 255  # [0, 1]
    return img_np


# def image_to_numpy_normalized(img: Image.Image) -> np.ndarray:
#     r"""Converts a PIL RGB Image to NumPy array of range `[-1, 1]` in HWC format."""
#     img_np = image_to_numpy_unnormalized(img)  # [0, 1]
#     img_np *= 2  # [0, 2]
#     img_np -= 1  # [-1, 1]
#     return img_np


class FullTransform:
    def __call__(self, data: dict[str, Image.Image]):
        lq = image_to_numpy_unnormalized(data['lq_image'])
        hq = image_to_numpy_unnormalized(data['hq_image'])

        H, W, _ = lq.shape
        lq = lq[:H // 16 * 16, :W // 16 * 16]
        hq = hq[:H // 16 * 16, :W // 16 * 16]

        data['lq'] = torch.from_numpy(np.ascontiguousarray(lq)).permute(2, 0, 1).float()
        data['gt'] = torch.from_numpy(np.ascontiguousarray(hq)).permute(2, 0, 1).float()

        return data


class TrainTransform:
    def __init__(
        self,
        crop_size: tuple[int, int] | int,
        resize: bool = False,
        flip_rotate: bool = False,
    ):
        self.crop_size = to_2tuple(crop_size)
        
        if resize:
            self.resize = tv_trans.RandomResizedCrop(
                self.crop_size,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                interpolation=tv_trans.InterpolationMode.BICUBIC,
                antialias=True,
            )
        else:
            self.resize = None
        
        self.do_flip_rotate = flip_rotate
        
        logger.debug(f'TRAIN DATASET `resize={resize}`, `do_flip_rotate={self.do_flip_rotate}`')

    def random_resize_crop(self, image: np.ndarray, gt: np.ndarray, target_height: int, target_width: int):
        """Performs random crop, resizing if necessary."""
        if self.resize is None:
            height, width = image.shape[:2]
            old_shape = image.shape

            # resize image until short size is no less than target size
            resize_ratio = max(target_height / height, target_width / width)
            if resize_ratio > 1:
                new_size = (math.ceil(resize_ratio * width), math.ceil(resize_ratio * height))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
                gt = cv2.resize(gt, new_size, interpolation=cv2.INTER_CUBIC)
                height, width = image.shape[:2]

            try:
                x_start = np.random.randint(0, width - target_width + 1)
                y_start = np.random.randint(0, height - target_height + 1)
            except:
                logger.exception(
                    f'{old_shape=}, {height=}, {width=}, {target_height=}, {target_width=}', main_process_only=False
                )
                raise
            cropped_image = image[y_start:y_start+target_height, x_start:x_start+target_width]
            cropped_gt = gt[y_start:y_start+target_height, x_start:x_start+target_width]
            # To Tensor
            cropped_image = torch.from_numpy(cropped_image.transpose(2, 0, 1))
            cropped_gt = torch.from_numpy(cropped_gt.transpose(2, 0, 1))
        
        else:
            # To Tensor
            image = torch.from_numpy(image.transpose(2, 0, 1))
            gt = torch.from_numpy(gt.transpose(2, 0, 1))
            # random resized crop
            cropped_image, cropped_gt = self.resize(torch.stack([image, gt])).unbind(0)

        return cropped_image, cropped_gt
    
    @staticmethod
    def flip_rotate(image: torch.Tensor, gt: torch.Tensor):
        # Flip z-axis with 0.5 prob.
        if np.random.rand() < 0.5:
            tv_f.hflip(image)
            tv_f.hflip(gt)
        # Rotate with 1/4 prob for each angle.
        if np.random.rand() < 0.75:
            angle = int(np.random.choice([90, 180, 270]))
            image = tv_f.rotate(image, angle, expand=True)
            gt = tv_f.rotate(gt, angle, expand=True)
        return image, gt

    def __call__(self, data: dict[str, Image.Image]):
        # Summary: randomly resize & crop and convert PIL Image img/gt to CHW tensor in [0, 1]
        lq_image = image_to_numpy_unnormalized(data['lq_image'])
        hq_image = image_to_numpy_unnormalized(data['hq_image'])
        try:
            data['lq'], data['gt'] = self.random_resize_crop(lq_image, hq_image, *self.crop_size)
            if self.do_flip_rotate:
                data['lq'], data['gt'] = self.flip_rotate(data['lq'], data['gt'])
        except:
            logger.exception(
                f"{self.crop_size=}, {data['lq_image'].size=}, {data['hq_image'].size=}, {lq_image.shape=}, {hq_image.shape=}",
                main_process_only=False,
            )
            raise

        return data


class TestTransform:
    def __init__(
        self,
        crop_size: tuple[int, int] | int,
        overlap: int,
        resize: str = 'none',  # resize or crop strategy.
    ):
        assert crop_size is not None
        self.crop_size = to_2tuple(crop_size)
        self.overlap = overlap
        self.resize = resize

    def resize_or_crop(self, img: Image.Image) -> tuple[list[Image.Image], Image.Image]:
        if self.resize == 'none':
            ratio = max(img.height, img.width) / 1920
            if ratio > 1:
                img = img.resize((round(img.size[0] / ratio), round(img.size[1] / ratio)), resample=Image.Resampling.BICUBIC)
            crops = [img]  # exactly one crop
        elif self.resize == 'resize':
            img = img.resize(self.crop_size, resample=Image.Resampling.BICUBIC)
            crops = [img]  # exactly one crop
        elif self.resize == 'crop':
            # resize image until short size is no less than `crop_size`
            target_width, target_height = self.crop_size
            width, height = img.size
            resize_ratio = max(target_height / height, target_width / width)
            if resize_ratio > 1:
                new_size = (math.ceil(resize_ratio * width), math.ceil(resize_ratio * height))
                img = img.resize(new_size, resample=Image.Resampling.BICUBIC)
            crops = sliding_window_crop(img, self.crop_size, self.overlap)
        elif self.resize == 'center_crop':
            # resize image until short size is no less than `crop_size`
            target_width, target_height = self.crop_size
            width, height = img.size
            resize_ratio = max(target_height / height, target_width / width)
            if resize_ratio > 1:
                new_size = (math.ceil(resize_ratio * width), math.ceil(resize_ratio * height))
                img = img.resize(new_size, resample=Image.Resampling.BICUBIC)
            # Adapted from [Restormer](https://github.com/swz30/Restormer/blob/df766d5521afd21cce56e24159f300b30ebf3fb0/Motion_Deblurring/generate_patches_gopro.py#L58C5-L64C64)
            width, height = img.size
            left = (width - target_width)//2
            upper = (height - target_height)//2
            img = img.crop([left, upper, left+target_width, upper+target_height])
            crops = [img]
        else:
            raise NotImplementedError(f'Unsupported transform option {self.resize}')
        
        crops = [torch.from_numpy(image_to_numpy_unnormalized(c).transpose(2, 0, 1)) for c in crops]
        
        return crops, img

    def __call__(self, data: dict[str, Image.Image]):
        data['lq'], data['lq_image'] = self.resize_or_crop(data['lq_image'])
        data['gt'], data['hq_image'] = self.resize_or_crop(data['hq_image'])
        return data


def make_dataset(
    dataset: str,
    split: str = 'train',
    transform: Optional[str] = None,
    crop_size: tuple[int, int] | int = (512, 512),
    overlap: int = 0,
    train_resize: bool = False,
    train_flip_rotate: bool = False,
    test_resize: str = 'none',
    **kwargs,  # passed to `GIRDataset.__init__`
) -> GIRDataset:
    r"""Makes a dataset instance."""

    if transform == 'train' or (transform is None and split == 'train'):
        transforms = TrainTransform(crop_size, train_resize, train_flip_rotate)
    elif transform == 'test' or (transform is None and split == 'test'):
        transforms = TestTransform(crop_size, overlap, test_resize)
    elif transform == 'test_full':
        transforms = FullTransform()
    else:
        raise ValueError(f'cannot find transform for `{split=}` and `{transform=}`')

    dataset = str(dataset).lower()
    datacls = DATASETS.get(dataset, None)
    if datacls is None:
        raise ValueError(f'unknown dataset {dataset}; expetcs one of {set(DATASETS)}')

    # NOTE: some datasets (proof) does not have `annotations`, so do not check its exististence
    annotations = os.path.join('configs', 'data', f'{datacls.__name__}_{split}.json')
    data = datacls(annotations, split, transforms, **kwargs)
    return data


def flatten_crops(crops_list: list[list[T]]) -> tuple[list[T], list[int]]:
    num_crops = [len(crops) for crops in crops_list]
    flattened_crops = [c for crops in crops_list for c in crops]
    return flattened_crops, num_crops


def unflatten_crops(flattened_crops: list[T], num_crops: list[int]) -> list[list[T]]:
    cumsum_num_crops = [sum(num_crops[:i]) for i in range(len(num_crops) + 1)] # [0, ...]
    assert cumsum_num_crops[-1] == len(flattened_crops), (cumsum_num_crops, len(flattened_crops))
    crops_list = [flattened_crops[start:end] for start, end in zip(cumsum_num_crops, cumsum_num_crops[1:])]
    return crops_list


def postprocess_crops(
    flattened_crops: list[T],
    num_crops: list[int],
    original_sizes: list[tuple[int, int]],
    crop_size: tuple[int, int] | int,
    overlap: int,
) -> list[T]:
    r"""Post-processes flattned crops by unflattening and merging them."""
    crops_list = unflatten_crops(flattened_crops, num_crops)
    crop_size = to_2tuple(crop_size)
    images = [
        merge_crops(crops, original_size, crop_size, overlap)
        for crops, original_size in zip(crops_list, original_sizes)
    ]
    return images


def get_test_collate_fn():
    r"""Gets a collate function for testing."""
    # Collates crops by concatenating them into a list, and inserting
    # `num_crops` to separate the crops.
    def collate(batch: list[dict]) -> dict:
        # Pop and flatten `lq` lists.
        lq = [d.pop('lq') for d in batch]
        lq, num_crops = flatten_crops(lq)
        gt = [d.pop('gt') for d in batch]
        gt, num_crops_hq = flatten_crops(gt)
        assert num_crops == num_crops_hq, (num_crops, num_crops_hq)
        batch = torch.utils.data.default_collate(batch)
        batch['lq'] = torch.stack(lq)
        batch['gt'] = torch.stack(gt)
        batch['num_crops'] = num_crops
        return batch

    return collate
