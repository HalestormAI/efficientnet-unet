import torch

import dataset.voc_dataset as voc_dataset
from dataset.segmentation_utils import (
    Compose,
    Resize, CenterCrop, PILToTensor, ConvertImageDtype, UncertaintyToBackground
)

_DATASET = {}


def get_transforms(img_size):
    return Compose([
        Resize(img_size),
        CenterCrop([img_size, img_size]),
        PILToTensor(),
        UncertaintyToBackground(),
        ConvertImageDtype(torch.float),
    ])


def load_dataset(args):
    transform = get_transforms(args.image_size)
    if args.dataset.startswith("voc"):
        _DATASET["train"], _DATASET["eval"] = voc_dataset.get_dataset(args.dataset, transform)
        _DATASET["num_classes"] = voc_dataset.NUM_CLASSES
    else:
        raise NotImplementedError("Only V0C dataset is implemented so far")


def train(args):
    if "train" not in _DATASET:
        load_dataset(args)
    return _DATASET["train"]


def eval(args):
    if "eval" not in _DATASET:
        load_dataset(args)
    return _DATASET["eval"]


def num_classes(args):
    if "num_classes" not in _DATASET:
        load_dataset(args)
    return _DATASET["num_classes"]
