import os.path

from torchvision.datasets import VOCSegmentation

NUM_CLASSES = 21


def get_dataset(dataset_name, transform):
    ds_year = dataset_name[-4:]

    download = not os.path.isdir(f"data/voc_{ds_year}")

    train_dataset = VOCSegmentation(f"data/voc_{ds_year}", image_set="train", year=ds_year, transforms=transform,
                                    download=download)
    eval_dataset = VOCSegmentation(f"data/voc_{ds_year}", image_set="val", year=ds_year, transforms=transform)
    return train_dataset, eval_dataset
