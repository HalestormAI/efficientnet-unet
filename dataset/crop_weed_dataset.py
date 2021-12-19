import yaml
from PIL import Image
from torch.utils.data import Dataset

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class CropWeedFieldImageDataset(Dataset):
    def __init__(self, root_path: str):
        super().__init__()

        self.root_path = root_path

    def __len__(self):
        return


if __name__ == "__main__":
    ann_path = "/Users/ianhales/PycharmProjects/udock/data/dataset-1.0/annotations/001_annotation.yaml"
    img_path = "/Users/ianhales/PycharmProjects/udock/data/dataset-1.0/images/001_image.png"

    img = Image.open(img_path)

    with open(ann_path) as fh:
        data = yaml.load(fh, Loader=Loader)

    colours = {
        "weed": 1,
        "crop": 2
    }

    for a in data["annotation"]:
        pass
