import argparse

from model.loss import LossType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Application parameters")
    group.add_argument("--mode", choices=("train", "infer"), required=True)
    group.add_argument("--dataset",
                       choices=("voc2007", "voc2008", "voc2009", "voc2010", "voc2011", "voc2012",),
                       required=True)

    group = parser.add_argument_group("Model parameters")
    group.add_argument("--batch-size", default=4, type=int)
    group.add_argument("--image-size", default=224, type=int)
    group.add_argument("--model-size", default=0, type=int, choices=tuple(range(8)))
    model_modifiers = group.add_mutually_exclusive_group()
    model_modifiers.add_argument("--remove-batchnorm",
                                 action="store_true",
                                 help="Replace batchnorms in the backbone with identity. Note this will only work with "
                                      "untrained models.")
    model_modifiers.add_argument("--pretrained-backbone",
                                 action="store_true",
                                 help="Load the imagenet model weights from torchvision for the backbone.")

    group = parser.add_argument_group("Training parameters")
    group.add_argument("--learning-rate", default=0.001, type=float)
    group.add_argument("--epochs", default=10, type=int)
    group.add_argument("--losses",
                       nargs="+",
                       default=("jaccard", "cce"),
                       type=LossType,
                       choices=tuple(e for e in LossType))
    return parser.parse_args()
