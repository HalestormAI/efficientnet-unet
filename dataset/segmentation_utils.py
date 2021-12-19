from typing import List, Union

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

# The standard torchvision transforms take and return a single tensor. For segmentation we typically want to transform
# the image and target together (as in the VOC Dataset), These helper functions apply equivalent transorms (in most
# cases) to the image and target.
# Mostly taken from / inspired by
# https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, size: Union[int, List[int]], interpolation=T.InterpolationMode.NEAREST):
        self.size = size
        self.interpolation_mode = interpolation

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=self.interpolation_mode)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    """Convert the image to the dtype. Ignores the labels as we typically want to keep them in int64"""
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class UncertaintyToBackground:
    def __init__(self, uncertain_class: int = 255, bg_class: int = 0):
        self.bg_class = bg_class
        self.uncertain_class = uncertain_class

    def __call__(self, image, target):
        target[target == self.uncertain_class] = self.bg_class
        return image, target
