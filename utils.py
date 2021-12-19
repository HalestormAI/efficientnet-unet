import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Application parameters")
    group.add_argument("--mode", choices=("train", "infer"), required=True)
    group.add_argument("--dataset",
                       choices=("voc2007", "voc2008", "voc2009", "voc2010", "voc2011", "voc2012",),
                       required=True)

    group = parser.add_argument_group("Model parameters")
    group.add_argument("--batch-size", default=8, type=int)
    group.add_argument("--image-size", default=384, type=int)
    group.add_argument("--model-size", default=4, type=int, choices=tuple(range(8)))

    group = parser.add_argument_group("Training parameters")
    group.add_argument("--epochs", default=20, type=int)  # TODO: is this a sensible default?
    return parser.parse_args()
