from __future__ import annotations

import os
from typing import Callable, Any, Optional

import numpy as np
from imgaug import augmenters as iaa
from pathlib import Path
from PIL import Image, ImageFilter

from basicsr.datasets.bases import GIRDataset
from basicsr.datasets.utils import random_split
from basicsr.transforms.distortion_aug import blur_related


class proof(GIRDataset):
    r"""Proof-of-concept dataset that returns nothing but gaussian noises."""
    def __init__(
        self,
        annotations: str | list[dict[str, str]],
        split: str,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        self.transform = transform
        self.annotations: list[dict[str, str]] = self.make_annotations(None, split)
        self._task = self._get_task(self.annotations[0]['lq'])

        if kwargs.get('cached_latent_root', None) is not None:
            raise ValueError('proof does not support caching')

    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        num_samples = 64 if split == 'train' else 16
        return [
            {'lq': os.path.join('All_in_one', 'Gaussian_denoising', 'lq')},
            {'hq': os.path.join('All_in_one', 'Gaussian_denoising', 'hq')},
        ] * num_samples

    def load_lq_image(self, path: str) -> Image.Image:
        image = np.random.normal(0, 0.1, (512, 512, 3))
        image = Image.fromarray(image, 'RGB')
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
        return image
    
    def load_hq_image(self, path: str) -> Image.Image:
        image = Image.new('RGB', (512, 512), 0)
        return image


class dpdd(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        root = os.path.join(root, 'train_c' if split == 'train' else 'test_c')

        sample_path = os.path.join(root, 'source')
        gt_path = os.path.join(root, 'target')

        annotations = []
        for sample, gt in zip(os.listdir(sample_path), os.listdir(gt_path)):
            annotations.append({
                'lq': os.path.join(sample_path, sample),
                'hq': os.path.join(gt_path, gt),
            })
        return annotations


class dense_haze(GIRDataset): # 只有train数据集
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        train_data_list_path = os.path.join(root, 'hazy')
        GT_data_list_path = os.path.join(root, 'GT')

        annotations = []
        for file in os.listdir(train_data_list_path):
            annotations.append({
                'lq': os.path.join(train_data_list_path, file),
                'hq': os.path.join(GT_data_list_path, file.replace('hazy', 'GT')),
            })
        annotations = random_split(annotations, split=split)
        return annotations


class nh_haze(GIRDataset): # 只有train数据集
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        annotations = []
        for file in os.listdir(root):
            if file.endswith('hazy.png'):
                annotations.append({
                    'lq': os.path.join(root, file),
                    'hq': os.path.join(root, file.replace('hazy', 'GT')),
                })
        annotations = random_split(annotations, split=split)
        return annotations


class sots(GIRDataset): # 只有train
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        if not os.path.isdir(os.path.join(root, 'indoor', 'gt_clean')):
            raise NotADirectoryError('please run `labs/cleandata_sots.py` to clean SOTS')
        annotations = []
        for task in ['indoor', 'outdoor']:
            sample_path = os.path.join(root, task, 'hazy')
            gt_path = os.path.join(root, task, 'gt_clean' if task == 'indoor' else 'gt')
            for file in os.listdir(sample_path):
                annotations.append({
                    'lq': os.path.join(sample_path, file),
                    'hq': os.path.join(gt_path, file.split('_')[0] + '.png'),
                    'subset': task,
                })
        annotations = random_split(annotations, split=split)
        return annotations


class outdoor_rain(GIRDataset): # 只有train
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        annotations = []

        sample_path = os.path.join(root, 'in')
        gt_path = os.path.join(root, 'gt')

        for file in os.listdir(sample_path):
            annotations.append({
                'lq': os.path.join(sample_path, file),
                'hq': os.path.join(gt_path, '{}_{}.png'.format(*file.split('_')[:2])),
            })
        annotations = random_split(annotations, split=split)
        return annotations


class rain1400(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        root = os.path.join(root, 'training' if split == 'train' else 'testing')
        annotations = []

        sample_path = os.path.join(root, 'rainy_image')
        gt_path = os.path.join(root, 'ground_truth')

        for file in os.listdir(sample_path):
            annotations.append({
                'lq': os.path.join(sample_path, file),
                'hq': os.path.join(gt_path, '{}.jpg'.format(file.split('_')[0])),
            })
        return annotations


class lhp(GIRDataset):
    # TODO: LHP has train split, but some images (e.g. `2052.png`)
    # have mismatched LQ and HQ sizes. Until we find a workaround,
    # it is unadvisable to use it for training.
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        assert split == 'test', 'LHP is unsuitable for training'
        annotations = []

        sample_path = os.path.join(root, 'input', split)
        gt_path = os.path.join(root, 'gt', split)

        for file in os.listdir(sample_path):
            annotations.append({
                'lq': os.path.join(sample_path, file),
                'hq': os.path.join(gt_path, file),
            })
        return annotations


class realsnow(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        root = os.path.join(root, 'training' if split == 'train' else 'testing')
        annotations = []

        if split == 'train':
            sample_path = os.path.join(root, 'video2imgs_IN_re_patches')
            gt_path = os.path.join(root, 'video2imgs_GT_re_patches')
        else:
            sample_path = os.path.join(root, 'video2imgs_IN_testing_re')
            gt_path = os.path.join(root, 'video2imgs_GT_testing_re')

        for file in os.listdir(sample_path):
            annotations.append({
                'lq': os.path.join(sample_path, file),
                'hq': os.path.join(gt_path, file),
            })
        return annotations


class snow100k(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        root = os.path.join(root, 'train' if split == 'train' else 'test')
        annotations = []

        if split == 'train':
            sample_path = os.path.join(root, 'synthetic')
            gt_path = os.path.join(root, 'gt')
            for file in os.listdir(sample_path):
                annotations.append({
                    'lq': os.path.join(sample_path, file),
                    'hq': os.path.join(gt_path, file),
                })
        else:
            for task in ['Snow100K-S', 'Snow100K-M', 'Snow100K-L']:
                sample_path = os.path.join(root, task, 'synthetic')
                gt_path = os.path.join(root, task, 'gt')
                for file in os.listdir(sample_path):
                    annotations.append({
                        'lq': os.path.join(sample_path, file),
                        'hq': os.path.join(gt_path, file),
                        'subset': task,
                    })

        return annotations


class lol_v2(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        annotations = []

        for task in ['Real_captured', 'Synthetic']:
            sample_path = os.path.join(root, task, 'Train' if split == 'train' else 'Test', 'Low')
            gt_path = os.path.join(root, task, 'Train' if split == 'train' else 'Test', 'Normal')
            for file in os.listdir(sample_path):
                annotations.append({
                    'lq': os.path.join(sample_path, file),
                    'hq': os.path.join(gt_path, file.replace('low', 'normal')),
                    'subset': task,
                })

        return annotations


class gopro(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        root = os.path.join(root, 'train' if split == 'train' else 'test')
        annotations = []

        for img_dir in os.listdir(root):
            in_sample_path = os.path.join(root, img_dir, 'blur_gamma')
            gt_sample_path = os.path.join(root, img_dir, 'sharp')

            for img in os.listdir(in_sample_path):
                annotations.append({
                    'lq': os.path.join(in_sample_path, img),
                    'hq': os.path.join(gt_sample_path, img),
                })

        return annotations


class gopro_nongamma(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        raise NotImplementedError

    def _get_task(self, path: str) -> str:
        return "Motion_deblurring"


class raindrop(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        root = os.path.join(root, 'train' if split == 'train' else '')
        annotations = []

        if split == 'train':
            sample_path = os.path.join(root, 'data')
            gt_path = os.path.join(root, 'gt')
            for file in os.listdir(sample_path):
                annotations.append({
                    'lq': os.path.join(sample_path, file),
                    'hq': os.path.join(os.path.join(gt_path, file.replace('rain', 'clean'))),
                })
        else:
            for task in ['test_a', 'test_b']:
                sample_path = os.path.join(root, task, 'data')
                gt_path = os.path.join(root, task, 'gt')
                for file in os.listdir(sample_path):
                    annotations.append({
                        'lq': os.path.join(sample_path, file),
                        'hq': os.path.join(os.path.join(gt_path, file.replace('rain', 'clean'))),
                        'subset': task,
                    })

        return annotations


class rainds(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        annotations = []

        # for task in ['RainDS_real', 'RainDS_syn']:
        for task in ['RainDS_real']:
            sample_path = os.path.join(root, task, 'train_set' if split == 'train' else 'test_set', 'raindrop')
            gt_path = os.path.join(root, task, 'train_set' if split == 'train' else 'test_set', 'gt')
            for file in os.listdir(sample_path):
                if file.endswith('.png'):
                    annotations.append({
                        'lq': os.path.join(sample_path, file),
                        'hq': os.path.join(gt_path, file),
                        'subset': task,
                    })

        return annotations


class sidd(GIRDataset): # 只有train
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        root = os.path.join(root, 'Data')
        annotations = []

        for img_dir in os.listdir(root):
            for img in os.listdir(os.path.join(root, img_dir)):
                if 'NOISY' in img:
                    annotations.append({
                        'lq': os.path.join(root, img_dir, img),
                        'hq': os.path.join(root, img_dir, img.replace('NOISY', 'GT')),
                    })

        annotations = random_split(annotations, split=split)
        return annotations


class rain100h(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        annotations = []

        for task in ['Heavy', 'Light']:
            sample_path = os.path.join(root, 'rain_data_'+('train_' if split == 'train' else 'test_')+task, 'rain', 'X2')
            gt_path = os.path.join(root, 'rain_data_'+('train_' if split == 'train' else 'test_')+task, 'norain')
            for file in os.listdir(sample_path):
                annotations.append({
                    'lq': os.path.join(sample_path, file),
                    'hq': os.path.join(gt_path, file.replace('x2.png', '.png')),
                    'subset': task,
                })

        return annotations


class rain100ho(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        annotations = []

        root = Path(root) / ('RainTrainH' if split == 'train' else 'Rain100H')
        norain_paths =  list(filter(lambda p: p.stem.startswith('norain-'), root.glob('*.png')))

        for norain_path in norain_paths:
            index = norain_path.stem.split('-')[1]
            rain_path = norain_path.parent / f'rain-{index}.png'
            assert rain_path.is_file(), rain_path
            annotations.append({'lq': str(rain_path), 'hq': str(norain_path)})

        return annotations


class rain100l(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        annotations = []

        sample_path = os.path.join(root, 'RainTrainL' if split == 'train' else 'Rain100L', 'rainy')
        gt_path = os.path.join(root, 'RainTrainL' if split == 'train' else 'Rain100L', 'norain')
        for file in os.listdir(sample_path):
            annotations.append({
                'lq': os.path.join(sample_path, file),
                'hq': os.path.join(gt_path, file.replace('rain', 'norain')),
            })

        return annotations


class reside_its(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        annotations = []

        sample_path = os.path.join(root, 'hazy')
        gt_path = os.path.join(root, 'clear')
        for file in os.listdir(sample_path):
            annotations.append({
                'lq': os.path.join(sample_path, file),
                'hq': os.path.join(gt_path, file.split('_')[0] + '.png'),
            })

        annotations = random_split(annotations, split=split)
        return annotations


class realblur(GIRDataset):
    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        annotations = []

        for task in ['J', 'R']:
            file_name = 'RealBlur_'+task+('_train' if split == 'train' else '_test')+'_list.txt'
            with open(os.path.join(root, file_name), 'r') as file:
                for line in file:
                    annotations.append({
                        'lq': os.path.join(root, line.rstrip().split(' ')[1]),
                        'hq': os.path.join(root, line.rstrip().split(' ')[0]),
                        'subset': task,
                    })

        return annotations


class wed(GIRDataset):
    def __init__(
        self,
        annotations: str | list[dict[str, str]],
        split: str,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(annotations, split, transform, **kwargs)
        self.add_gaussian_noise = iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255), per_channel=True)

    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        annotations = []

        sample_path = os.path.join(root, 'ref', 'all')
        gt_path = os.path.join(root, 'ref', 'all')
        for file in os.listdir(sample_path):
            annotations.append({
                'lq': os.path.join(sample_path, file),
                'hq': os.path.join(gt_path, file),
            })

        annotations = random_split(annotations, split=split)
        return annotations

    def load_lq_image(self, path: str) -> Image.Image:
        image = super().load_lq_image(path)
        image = self.add_gaussian_noise(image=np.asarray(image))
        image = Image.fromarray(image)
        return image


class wedblur(GIRDataset):
    def __init__(
        self,
        annotations: str | list[dict[str, str]],
        split: str,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(annotations, split, transform, **kwargs)
        self.add_gaussian_blur = blur_related

    def make_annotations(self, root: str, split: str) -> dict[str, str]:
        annotations = []

        sample_path = os.path.join(root, 'ref', 'all')
        gt_path = os.path.join(root, 'ref', 'all')
        for file in os.listdir(sample_path):
            annotations.append({
                'lq': os.path.join(sample_path, file),
                'hq': os.path.join(gt_path, file),
            })

        annotations = random_split(annotations, split=split)
        return annotations

    def load_lq_image(self, path: str) -> Image.Image:
        image = super().load_lq_image(path)
        image = self.add_gaussian_blur(image=np.asarray(image))
        image = Image.fromarray(image)
        return image


# class urban100(data.Dataset):
#     def __init__(self, root, index, transform, istrain=True):
#         if istrain:
#         imgpath = os.path.join(root, 'images')
#         csv_file = os.path.join(root, 'dmos.csv')
#         sample, gt = [], []
#         with open(csv_file) as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 if (int(row['dist_img'].split('_')[0].replace('I', '')) - 1) in index:
#                     sample.append(os.path.join(imgpath, row['dist_img']))
#                     mos = np.array(float(row['dmos'])).astype(np.float32)
#                     gt.append(mos)
#         self.samples_p, self.gt_p = sample, gt
#         self.transform = transform

#     def __getitem__(self, index):
#         img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)
#         return img_tensor, gt_tensor, img_size, "", 'kadid'

#     def __len__(self):
#         length = len(self.samples)
#         return length


DATASETS: dict[str, GIRDataset] = {
    'proof': proof,
    'dpdd': dpdd,
    # 'urban100': urban100,
    'dense-haze': dense_haze,
    'nh-haze': nh_haze,
    'sots': sots,
    'outdoor-rain': outdoor_rain,
    'rain1400': rain1400,
    'lhp': lhp,
    'realsnow': realsnow,
    'snow100k': snow100k,
    'lol-v2': lol_v2,
    'gopro': gopro,
    'gopro-nongamma': gopro_nongamma,
    'raindrop': raindrop,
    'rainds': rainds,
    'sidd': sidd,
    'rain100h': rain100h,
    'rain100ho': rain100ho,
    'rain100l': rain100l,
    'reside-its': reside_its,
    'realblur': realblur,
    'wed': wed,
    'wedblur': wedblur,
}
