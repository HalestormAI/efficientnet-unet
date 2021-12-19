import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary

import dataset
from model import EffUnet
from model.loss import DiceCoefficientLoss
from utils import parse_args

if __name__ == "__main__":
    args = parse_args()

    train_dataset = dataset.train(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    img, smnt = next(iter(train_dataloader))

    num_classes = dataset.num_classes(args)

    model = EffUnet(args.model_size, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceCoefficientLoss()

    x = model(img)

    summary(model, input_size=img.shape)
    print(x.shape)

    input_names = ["input_image"]
    output_names = ["output_logits"]
