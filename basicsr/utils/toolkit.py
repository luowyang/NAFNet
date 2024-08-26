from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename=None):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor, (C, H, W) or (1, C, H, W)
    :param filename: 保存的文件名
    """
    if input_tensor.ndim == 4:
        assert input_tensor.shape[0] == 1, input_tensor.shape
        input_tensor = input_tensor.squeeze(0)
    assert input_tensor.ndim == 3, input_tensor.shape
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.cpu().float()
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    if filename is not None:
        cv2.imwrite(filename, input_tensor)
    else:
        return input_tensor


@torch.no_grad()
def gather_together(data):  # 封装成一个函数，用于收集各个gpu上的data数据，并返回一个list
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def get_avg(self):
        return self.avg

    def format_values(self) -> str:
        fmtstr = "{val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def _image_to_numpy(image: Image.Image | np.ndarray | torch.Tensor) -> np.ndarray:
    old_type = type(image)
    # Converts image to ndarray of HWC or HW format.
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.ndim == 3:
            image = image.permute(1, 2, 0)  # CHW -> HWC
    if not isinstance(image, np.ndarray):
        raise TypeError(f'expects one of PIL Image, ndarray, tensor; gets {old_type}')
    if image.ndim < 2 or image.ndim > 3:
        raise ValueError(f'expects 2 or 3 dimensional image; gets {image.shape}')
    return image


def compute_psnr_ssim(
    predict: Image.Image | np.ndarray | torch.Tensor,
    target: Image.Image | np.ndarray | torch.Tensor,
    *,
    data_range: Optional[float] = None,
):
    r"""Computes PSNR and SSIM given a pair of images `predict` and `target`.
    
    Images can be one of:
    - PIL Image: RGB or grayscale image
    - ndarray: HWC or HW format
    - tensor: CHW or HW format

    For multi-channel images, PSNR is computed along all elements, SSIM is averaged
    along channels.

    Float images must specify `data_range`, see documentation of `skimage.metrics.structural_similarity`.
    """
    predict = _image_to_numpy(predict)
    target = _image_to_numpy(target)
    assert predict.shape == target.shape, (predict.shape, target.shape)
    ssim_res = ssim(predict, target, data_range=data_range, channel_axis=(2 if predict.ndim == 3 else None))
    psnr_res = psnr(predict, target)

    return psnr_res, ssim_res


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def format(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def image_grid(
    imgs: Sequence[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    *,
    strict: bool = True,
    resize: bool = True,
) -> Image.Image:
    r"""Creates an image grid of `(rows, cols)` PIL images.

    Args:
        imgs: Sequence of PIL images.
        rows (optional): Number of rows. If `rows` is `None`, then `rows=len(imgs)//cols`; if
            `cols` is also `None`, then `rows=1` and `cols=len(imgs)`.
        cols (optional): Number of columns. If `cols` is `None`, then `cols=len(imgs)//rows`;
            if `rows` is also `None`, then `rows=1` and `cols=len(imgs)`.
        strict (optional): If `True` (default), all images must have exactly the same size;
            otherwise the images are resized/padded to the max size in right and bottom sides.
        resize (optional): If `True` (default), images are resized to the max size, otherwise
            the images are padded to the max size in right and bottom sides.
    """
    # Check and compute `rows` and `cols`.
    if rows is None:
        rows = 1 if cols is None else len(imgs) // cols
    if cols is None:
        cols = len(imgs) // rows
    assert len(imgs) == rows * cols, f'{len(imgs)} != {rows} * {cols}'

    max_w, max_h = imgs[0].size
    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            raise TypeError(f'imgs[{i}] is {type(img)}; expects PIL Image')
        if strict:
            assert img.size == (max_w, max_h), (
                f'imgs[{i}].size is {img.size}; expects ({max_w}, {max_h})'
            )
        else:
            max_w = max(max_w, img.width)
            max_h = max(max_h, img.height)

    grid = Image.new("RGB", size=(cols * max_w, rows * max_h))

    for i, img in enumerate(imgs):
        if img.size != (max_w, max_h):
            assert not strict, f'should not happen: imgs[{i}]={img}, size={img.size}, expects ({max_w}, {max_h})'
            if resize:
                img = img.resize((max_w, max_h))
        grid.paste(img, box=((i % cols) * max_w, (i // cols) * max_h))

    return grid


def to_image(img: torch.Tensor | np.ndarray) -> Image.Image:
    """Converts tensor or ndarray to Image."""
    if isinstance(img, torch.Tensor):
        img: np.ndarray = img.cpu().numpy()
    if not isinstance(img, np.ndarray):
        raise TypeError(f'expects tensor or ndarray, gets {type(img)}')
    img = (img * 255).transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img


def project_image(
    img: torch.Tensor | np.ndarray,
    method: str = 'pca',
) -> Image.Image:
    r"""Projects a multi-channel tensor or ndarray to 3-channel image.
    
    Args:
        img: torch.Tensor | np.ndarray
            The tensor of shape (C, H, W) or ndarray of shape (H, W, C) to project.
            The channel dimension can be omited for single-channel `img`.
        method: str, defaults to `pca`
            Projection method, supported methods are `pca` and `avg`.
    
    Returns:
        PIL Image: Projected RGB image.
    """
    raise NotImplementedError


def test_psnr_ssim():
    img1 = np.random.random((224, 224, 5))
    img2 = np.random.random((224, 224, 5))
    print(compute_psnr_ssim(img1, img2, data_range=1))

    img1 = (np.random.random((224, 224, 5)).clip(0, 1) * 255).astype(np.uint8)
    img2 = (np.random.random((224, 224, 5)).clip(0, 1) * 255).astype(np.uint8)
    print(compute_psnr_ssim(img1, img2))


if __name__ == '__main__':
    test_psnr_ssim()
