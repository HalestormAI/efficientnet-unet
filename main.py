import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm
from torch.utils.data import DataLoader
from torchinfo import summary

import dataset
from model import EffUnet
from model.loss import DiceCoefficientLoss
from utils import parse_args

if __name__ == "__main__":
    args = parse_args()

    train_dataloader = DataLoader(dataset.train(args), batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset.eval(args), batch_size=args.batch_size, shuffle=True)

    num_classes = dataset.num_classes(args)

    model = EffUnet(args.model_size, num_classes=num_classes, activate_logits=False, remove_bn=True)
    summary(model, input_size=(args.batch_size, 3, args.image_size, args.image_size))

    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceCoefficientLoss(apply_sigmoid=True)

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(train_dataloader)):
            optimizer.zero_grad()
            logits = model(inputs)

            tgt_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
            cross_entropy_loss = criterion(logits, targets)
            dice = dice_loss(logits, tgt_one_hot, multiclass=True)
            total_loss = cross_entropy_loss

            total_loss.backward()
            optimizer.step()
            print(f"Loss: {total_loss:.5f} | CrossEntropy: {cross_entropy_loss:.5f} | DICE: {dice:.5f}")