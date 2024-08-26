from __future__ import annotations

from typing import Iterable, TypeVar

import numpy as np
from PIL import Image
import torch

ImageType = TypeVar('ImageType', Image.Image, torch.Tensor, np.ndarray)


def sliding_window_crop(
    image: Image.Image,
    crop_size: tuple[int, int],
    overlap: int,
) -> list[Image.Image]:
    """
    Apply sliding window crop on an image.

    Parameters:
    - image: PIL Image object.
    - crop_size: Tuple (width, height) of the crop size.
    - overlap: Integer for the overlap size between adjacent crops.

    Returns:
    - List of cropped Image objects.
    """
    crops = []
    width, height = image.size
    crop_width, crop_height = crop_size

    # Calculate step size based on overlap
    step_width = crop_width - overlap
    step_height = crop_height - overlap

    for x in range(0, width, step_width):
        for y in range(0, height, step_height):
            x1, x2 = x, x + crop_width
            y1, y2 = y, y + crop_height
            # If the crop box is out of borders,
            # move it to top-left until right inside borders.
            if x2 > width:
                x1 = width - crop_width
                x2 = width
            if y2 > height:
                y1 = height - crop_height
                y2 = height
            box = (x1, y1, x2, y2)
            crop = image.crop(box)
            crops.append(crop)

    return crops


def merge_crops(
    crops: Iterable[ImageType],
    original_size: tuple[int, int],
    crop_size: tuple[int, int],
    overlap: int,
) -> ImageType:
    r"""
    Args:
        crops: Crops (PIL Images, Tensor or ndarray) of a single image.
            Image is RGB, Tensor is (C, H, W), ndarray is (H, W, C)

    Returns:
        Merged image of the same type as `crops`'s elements.
    """
    crop_width, crop_height = crop_size

    step_width = crop_width - overlap
    step_height = crop_height - overlap

    if isinstance(crops[0], Image.Image):
        merged_image = Image.new('RGB', original_size)
        i = 0
        for x in range(0, original_size[0], step_width):
            for y in range(0, original_size[1], step_height):
                x1, y1 = x, y
                # Undo the crop box moving in `sliding_window_crop()`.
                if x + crop_width > original_size[0]:
                    x1 = original_size[0] - crop_width
                if y + crop_height > original_size[1]:
                    y1 = original_size[1] - crop_height
                merged_image.paste(crops[i], (x1, y1))
                i += 1

    elif isinstance(crops[0], torch.Tensor):
        with torch.no_grad():
            merged_image = torch.zeros(
                crops[0].size(0), original_size[1], original_size[0], 
                dtype=crops[0].dtype, device=crops[0].device,
            )
            i = 0
            for x in range(0, original_size[0], step_width):
                for y in range(0, original_size[1], step_height):
                    x1, y1 = x, y
                    if x + crop_width > original_size[0]:
                        x1 = original_size[0] - crop_width
                    if y + crop_height > original_size[1]:
                        y1 = original_size[1] - crop_height
                    try:
                        merged_image[:, y1:y1+crop_height, x1:x1+crop_width] = crops[i]
                    except:
                        print(merged_image.shape, merged_image[:, y1:y1+crop_height, x1:x1+crop_width].shape, crops[i].shape)
                        import sys; sys.stdout.flush()
                        raise
                    i += 1
    
    elif isinstance(crops[0], np.ndarray):
        merged_image = np.zeros(
            original_size[1], original_size[0], crops[0].shape[2],
            dtype=crops[0].dtype,
        )
        i = 0
        for x in range(0, original_size[0], step_width):
            for y in range(0, original_size[1], step_height):
                x1, y1 = x, y
                if x + crop_width > original_size[0]:
                    x1 = original_size[0] - crop_width
                if y + crop_height > original_size[1]:
                    y1 = original_size[1] - crop_height
                merged_image[y1:y1+crop_height, x1:x1+crop_width, :] = crops[i]
                i += 1
    
    else:
        raise TypeError(f'unknown crop type {type(crops[0])}')

    return merged_image


def merge_crops_with_average(
    crops: Iterable[Image.Image],
    original_size: tuple[int, int],
    crop_size: tuple[int, int],
    overlap: int,
) -> Image.Image:
    """
    Merge cropped images back into a single image with averaging in the overlapping areas.

    Parameters:
    - crops: List of PIL Image objects (cropped images).
    - original_size: Tuple (width, height) of the original image size.
    - crop_size: Tuple (width, height) of the crop size.
    - overlap: Integer for the overlap size between adjacent crops.

    Returns:
    - Merged PIL Image object.
    """
    merged_image = Image.new('RGB', original_size)
    crop_width, crop_height = crop_size

    # Calculate step size based on overlap
    step_width = crop_width - overlap
    step_height = crop_height - overlap

    i = 0
    for x in range(0, original_size[0], step_width):
        for y in range(0, original_size[1], step_height):
            if i < len(crops):
                x1, x2 = x, x + crop_width
                y1, y2 = y, y + crop_height
                if x2 > original_size[0]:
                    x1 = original_size[0] - crop_width
                if y2 > original_size[1]:
                    y1 = original_size[1] - crop_height

                box = (x1, y1, x2, y2)
                # Extract the corresponding area from the merged image
                existing_area = merged_image.crop(box)

                # If the existing area is not empty (has been previously pasted)
                if existing_area.getbbox():
                    # Average the overlapping area
                    averaged_area = Image.blend(existing_area, crops[i], alpha=0.5)
                    merged_image.paste(averaged_area, box)
                else:
                    # If it's the first time pasting in this area
                    merged_image.paste(crops[i], box)

                i += 1

    return merged_image


def plot_crops(crops):
    """
    Plot each cropped image on a matplotlib subplot.

    Parameters:
    - crops: List of PIL Image objects (cropped images).
    """
    import matplotlib.pyplot as plt
    num_crops = len(crops)
    cols = int(np.sqrt(num_crops))
    rows = (num_crops // cols) + (0 if num_crops % cols == 0 else 1)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten() if num_crops > 1 else [axes]

    for ax, crop in zip(axes, crops):
        ax.imshow(crop)
        ax.axis('off')

    # Hide any unused subplots
    for ax in axes[len(crops):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def test_crop_and_merge():
    image = Image.fromarray(
        np.random.randint(0, 256, (800, 1200, 3)).astype(np.uint8)
    )
    crops = sliding_window_crop(image, (256, 256), 0)
    merged_image = merge_crops(crops, image.size, (256, 256), 0)
    assert np.allclose(np.asarray(image), np.asarray(merged_image))


if __name__ == '__main__':
    test_crop_and_merge()
