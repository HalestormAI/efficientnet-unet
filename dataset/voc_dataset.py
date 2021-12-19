from torchvision.datasets import VOCSegmentation

NUM_CLASSES = 20


def get_dataset(dataset_name, transform):
    ds_year = dataset_name[-4:]
    train_dataset = VOCSegmentation(f"data/voc_{ds_year}", image_set="train", year=ds_year, transforms=transform)
    eval_dataset = VOCSegmentation(f"data/voc_{ds_year}", image_set="val", year=ds_year, transforms=transform)
    return train_dataset, eval_dataset
