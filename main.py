import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.datasets import VOCSegmentation

from dataset import get_transforms
from model import EffUnet

if __name__ == "__main__":
    batch_size = 2

    img_size = 384

    transform = get_transforms(img_size)

    train_dataset = VOCSegmentation("data/voc_2007", image_set="train", year="2007", transforms=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    mock_data = torch.rand((batch_size, 3, img_size, img_size), dtype=torch.float32)
    model = EffUnet(4, num_classes=3)

    x = model(mock_data)

    summary(model, input_size=(batch_size, 3, img_size, img_size))
    print(x.shape)

    input_names = ["input_image"]
    output_names = ["output_logits"]
