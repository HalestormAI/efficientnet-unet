import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "infer"))
    parser.add_argument("--dataset", choices=("voc2007", "voc2008", "voc2009", "voc2010", "voc2011", "voc2012",))
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--image-size", default=384, type=int)
    parser.add_argument("--model-size", default=4, type=int, choices=tuple(range(8)))
    return parser.parse_args()
