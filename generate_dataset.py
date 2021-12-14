#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a dataset of distorted qr codes
"""
from typing import List, Tuple
import qrcode
import numpy as np
import random
import math
from PIL import Image
from pathlib import Path
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import (
    RoundedModuleDrawer,
    GappedSquareModuleDrawer,
    CircleModuleDrawer,
    HorizontalBarsDrawer,
    SquareModuleDrawer,
)
from qrcode.image.styles.colormasks import (
    RadialGradiantColorMask,
    VerticalGradiantColorMask,
    HorizontalGradiantColorMask,
)
import hashlib


def pil_to_numpy(i):
    return np.array(i.getdata(), dtype="uint8").reshape(i.size[0], i.size[1], 3)


def random_text():
    size = int(math.exp(random.random() * math.log(1024)))
    return bytes(list(np.random.randint(0x20, 0x7E, size=size))).decode("ascii")


def random_qr() -> Tuple[StyledPilImage, List[List[bool]], str]:
    qr = qrcode.QRCode(
        version=1,
        error_correction=random.choice(
            (
                qrcode.constants.ERROR_CORRECT_H,
                qrcode.constants.ERROR_CORRECT_L,
                qrcode.constants.ERROR_CORRECT_M,
                qrcode.constants.ERROR_CORRECT_Q,
            )
        ),
        box_size=12,
        border=9,
    )
    text = random_text()
    qr.add_data(text)
    qr.make(fit=True)
    module_drawer = random.choice(
        (
            SquareModuleDrawer(),
            SquareModuleDrawer(),
            SquareModuleDrawer(),
            SquareModuleDrawer(),
            RoundedModuleDrawer(),
            GappedSquareModuleDrawer(),
            CircleModuleDrawer(),
            HorizontalBarsDrawer(),
        )
    )
    color_mask = random.choice(
        (
            None,
            None,
            None,
            None,
            None,
            VerticalGradiantColorMask(
                top_color=128 + np.random.random((3,)) * 128,
                bottom_color=np.random.random((3,)) * 255,
            ),
            RadialGradiantColorMask(
                center_color=128 + np.random.random((3,)) * 128,
                edge_color=128 + np.random.random((3,)) * 128,
            ),
            HorizontalGradiantColorMask(
                left_color=128 + np.random.random((3,)) * 128,
                right_color=np.random.random((3,)) * 255,
            ),
        )
    )

    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=module_drawer,
        **({"color_mask": color_mask} if color_mask else {}),
    )

    qr.border = 0
    matrix = qr.get_matrix()
    return img, matrix, text


from albumentations import (
    ShiftScaleRotate,
    CLAHE,
    Transpose,
    ShiftScaleRotate,
    Blur,
    OpticalDistortion,
    GridDistortion,
    HueSaturationValue,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    RandomBrightnessContrast,
    Flip,
    OneOf,
    Compose,
    CoarseDropout,
    ImageCompression,
)


def image_distortion(p=0.5):
    return Compose(
        [
            Flip(p=0.1),
            Transpose(p=0.1),
            CoarseDropout(
                max_holes=30,
                max_width=8,
                max_height=8,
                fill_value=np.random.randint(0, 255),
                p=0.2,
            ),
            GaussNoise(var_limit=(0.0, 500.0), p=0.75),
            OneOf(
                [
                    MotionBlur(blur_limit=19, p=0.7),
                    MedianBlur(blur_limit=19, p=0.7),
                    Blur(blur_limit=19, p=0.7),
                ],
                p=0.8,
            ),
            ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=8, p=0.6
            ),
            CoarseDropout(
                max_holes=3,
                max_width=50,
                max_height=5,
                fill_value=np.random.randint(0, 255),
                p=0.1,
            ),
            CoarseDropout(
                max_holes=3,
                max_width=5,
                max_height=50,
                fill_value=np.random.randint(0, 255),
                p=0.1,
            ),
            OpticalDistortion(distort_limit=0.001, p=0.5),
            GridDistortion(distort_limit=0.15, p=0.4),
            CLAHE(clip_limit=4, p=0.4),
            RandomBrightnessContrast(),
            HueSaturationValue(p=0.7),
            ImageCompression(quality_lower=0, quality_upper=90, p=0.5),
        ],
        p=p,
    )


def distort_image(img: StyledPilImage) -> Image.Image:
    image = pil_to_numpy(img)
    augmentation = image_distortion(p=0.9)
    augmented = augmentation(image=image)
    image = augmented["image"]
    return Image.fromarray(image).resize((256, 256))


def generate_image_pixels_and_text() -> Tuple[Image.Image, np.array, str]:
    img, matrix_list, text = random_qr()
    image = distort_image(img)
    matrix = np.array(matrix_list)
    return image, matrix, text


dataset_root = Path("dataset")
dataset_root.mkdir(parents=True, exist_ok=True)

def write_sample(id: str, image: Image.Image, matrix: np.array, text: str):
    print(f"Writing sample {id}")
    image.save(dataset_root / f"{id}.png")
    (dataset_root / f"{id}.txt").write_text(text)
    np.save(dataset_root / f"{id}.npy", matrix)


def create_sample(i: int):
    id = f"{i:06d}"
    if (dataset_root / f"{id}.png").exists():
        return
    image, matrix, text = generate_image_pixels_and_text()
    write_sample(id, image, matrix, text)


from multiprocessing import Pool


def parallel_generate_images(n: int):
    with Pool() as pool:
        pool.map(create_sample, range(n))


import sys


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    parallel_generate_images(n)


if __name__ == "__main__":
    main()
