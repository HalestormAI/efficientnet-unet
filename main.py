import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm
from torch.utils.data import DataLoader
from torchinfo import summary

import dataset
from model import EffUnet
from model.loss import DiceCoefficientLoss, JaccardLoss, get_losses
from utils import parse_args


def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, num_classes: int):
    model.eval()

    total_accuracy = 0
    total_dice = 0

    num_batches = len(dataloader)
    dice_loss = DiceCoefficientLoss(True)

    for inputs, targets in tqdm.tqdm(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            logits = model(inputs)

            sig_logits = torch.softmax(logits, dim=1)
            predictions = sig_logits.argmax(dim=1)
            num_correct_not_bg = (predictions[targets != 0] == targets[targets != 0]).sum()
            total_accuracy = num_correct_not_bg / torch.count_nonzero(targets != 0)
            pred_one_hot = F.one_hot(predictions, num_classes).permute(0, 3, 1, 2).float()

            total_dice += dice_loss(pred_one_hot, targets_one_hot, multiclass=True)

    model.train()
    return total_dice / num_batches, total_accuracy / num_batches


if __name__ == "__main__":
    args = parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(dataset.train(args), batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset.eval(args), batch_size=args.batch_size, shuffle=True)

    num_classes = dataset.num_classes(args)

    model = EffUnet(args.model_size,
                    num_classes=num_classes,
                    activate_logits=False,
                    remove_bn=args.remove_batchnorm,
                    load_weights=args.pretrained_backbone)
    summary(model, input_size=(args.batch_size, 3, args.image_size, args.image_size))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    _, loss_fns, loss_status_tpl = get_losses(args)

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(train_dataloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            losses = [fn(logits, targets, num_classes) for fn in loss_fns]
            total_loss = torch.stack(losses, dim=0).sum()

            total_loss.backward()
            optimizer.step()
            print(loss_status_tpl.format(total_loss, *losses))

        val_loss, val_accuracy = evaluate(model, val_dataloader, num_classes)
        print(f"Validation Loss: {val_loss:.5f}  |  Accuracy: {val_accuracy}")
