import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm
from torch.utils.data import DataLoader
from torchinfo import summary

import dataset
from model import EffUnet
from model.loss import DiceCoefficientLoss
from model.utils import logits_to_onehot
from utils import parse_args


def evaluate(model, dataloader, num_classes):
    model.eval()

    total_accuracy = 0
    total_dice = 0

    num_batches = len(dataloader)

    for inputs, targets in tqdm.tqdm(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            logits = model(inputs)
            pred_one_hot = logits_to_onehot(logits, num_classes)

            total_dice += dice_loss(pred_one_hot, targets_one_hot, multiclass=True)
            total_accuracy += (targets_one_hot == pred_one_hot).sum() / pred_one_hot.numel()

    model.train()
    return total_accuracy / num_batches, total_dice / num_batches


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(dataset.train(args), batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset.eval(args), batch_size=args.batch_size, shuffle=True)

    num_classes = dataset.num_classes(args)

    model = EffUnet(args.model_size, num_classes=num_classes, activate_logits=False, remove_bn=True)
    summary(model, input_size=(args.batch_size, 3, args.image_size, args.image_size))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceCoefficientLoss(apply_sigmoid=True)

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(train_dataloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            tgt_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
            cross_entropy_loss = criterion(logits, targets)
            dice = dice_loss(logits, tgt_one_hot, multiclass=True)
            total_loss = dice + cross_entropy_loss

            total_loss.backward()
            optimizer.step()
            print(f"Loss: {total_loss:.5f} | CrossEntropy: {cross_entropy_loss:.5f} | DICE: {dice:.5f}")

        evaluate(model, val_dataloader, num_classes)