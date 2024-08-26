from __future__ import annotations

import albumentations as A
import numpy as np
from imgaug import augmenters as iaa
from PIL import Image

# import Automold as am


__all__ = [
    'noise_related',
    'blur_related',
    'color_related',
    'other_task',
    'rainline_related',
    'snow_related',
    'fog_related',
    'motion_related',
    'defocus_related',
    'low_light_related',
    'compress_related',
]


noise_related = iaa.Sequential(
    [
        iaa.OneOf(
            [
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255), per_channel=True),
                iaa.AdditiveLaplaceNoise(scale=(0, 0.2 * 255)),
                iaa.AdditiveLaplaceNoise(scale=(0, 0.2 * 255), per_channel=True),
                iaa.AdditivePoissonNoise(lam=(0, 15)),
                iaa.AdditivePoissonNoise(lam=(0, 15), per_channel=True),
                iaa.MultiplyElementwise((0.5, 1.5)),
                iaa.MultiplyElementwise((0.5, 1.5), per_channel=1),
                iaa.ImpulseNoise((0, 0.1)),
                iaa.SaltAndPepper((0, 0.1)),
                iaa.SaltAndPepper((0, 0.1), per_channel=True),
            ]
        )
    ]
)


blur_related = iaa.Sequential(
    [
        iaa.OneOf(
            [
                iaa.GaussianBlur(sigma=(3.0, 9.0)),
                # iaa.AverageBlur(k=((5, 11), (1, 3))),
                # iaa.MedianBlur(k=(3, 11)),
                # iaa.BilateralBlur(
                #     d=(3, 10),
                #     sigma_color=(10, 250),
                #     sigma_space=(10, 250),
                # ),
                # iaa.MotionBlur(k=(3, 7)),
                # iaa.MeanShiftBlur(),
            ]
        )
    ]
)


color_related = iaa.Sequential(
    [
        iaa.OneOf(
            [
                iaa.MultiplyBrightness((0.5, 1.5)),
                iaa.MultiplyHue((0.5, 1.5)),
                iaa.MultiplySaturation((0.5, 1.5)),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.ChangeColorTemperature((1100, 10000)),
                iaa.GammaContrast((0.5, 2.0)),
            ]
        )
    ]
)


other_task = iaa.Sequential(
    iaa.OneOf(
        [
            iaa.Clouds(),
            iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03)),
            iaa.Rain(speed=(0.1, 0.9), drop_size=(0.1, 1.0)),
            iaa.Fog(),
        ]
    )
)


rainline_related = iaa.Sequential(
    iaa.OneOf(
        [
            iaa.Rain(speed=(0.1, 0.9), drop_size=(0.1, 1.0)),
        ]
    )
)


snow_related = iaa.Sequential(
    iaa.OneOf(
        [
            iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03)),
        ]
    )
)


fog_related = iaa.Sequential(
    iaa.OneOf(
        [
            iaa.Fog(),
        ]
    )
)


motion_related = iaa.Sequential(
    iaa.OneOf(
        [
            iaa.MotionBlur(),
        ]
    )
)


defocus_related = A.Compose([
    A.Defocus(),
])


low_light_related = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.8, 0), p=1),
])


compress_related = iaa.Sequential(
    iaa.JpegCompression(compression=(65, 99)),
)


if __name__ == '__main__':
    from lllm_dif.utils.toolkit import image_grid, compute_psnr_ssim

    hq = Image.open('assets/hq_test.png')
    hq = np.array(hq)

    lq = blur_related(image=hq)
    print(lq.shape, compute_psnr_ssim(lq, hq))

    images = image_grid([Image.fromarray(lq), Image.fromarray(hq)])

    images.save('assets/lq-hq_test.png')
