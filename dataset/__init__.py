import torch

from dataset.segmentation_utils import (
    Compose,
    Resize, CenterCrop, PILToTensor, ConvertImageDtype, UncertaintyToBackground
)


def get_transforms(img_size):
    return Compose([
        Resize(img_size),
        CenterCrop([img_size, img_size]),
        PILToTensor(),
        UncertaintyToBackground(),
        ConvertImageDtype(torch.float),
    ])
