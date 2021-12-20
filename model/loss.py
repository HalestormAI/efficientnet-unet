import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum


class LossType(Enum):
    CCE = "cce"
    Jaccard = "jaccard"
    DICE = "dice"


class DiceCoefficientLoss(nn.Module):
    def __init__(self, apply_softmax: bool = False, eps: float = 1e-6):
        super().__init__()

        self.apply_softmax = apply_softmax
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor, multiclass=True) -> torch.Tensor:
        """
        If we're doing multiclass segmentation, we want to calculate dice for each channel independently and then mean-
        reduce afterwards.
        :param x: The estimated segmentation logits
        :param y: The labels
        :param multiclass: Whether the logits should be calculated multiclass-wise.
        :return: The Dice score, averaged over channels if multiclass.
        """

        if x.size() != y.size():
            raise RuntimeError(
                f"Cannot calculate DICE score - input and label size do not match ({x.shape} vs. {y.shape})")

        dice = 0
        if multiclass:
            for cls_idx in range(x.shape[1]):
                # Slice the logits for this class and pass to the dice function
                dice += self._dice(x[:, cls_idx, ...], y[:, cls_idx, ...])
            dice = dice / x.shape[1]
        else:
            dice = self._dice(x, y)
        return 1 - dice

    def _dice(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the DICE score for input logits, x, against labels, y.
        :param x: The estimated segmentation logits
        :param y: The labels
        :return: The dice score for this pair
        """
        if self.apply_softmax:
            x = torch.softmax(x, dim=1)

        x = x.view(-1)
        y = y.view(-1)

        intersection = torch.dot(x, y)
        return (2. * intersection + self.eps) / (x.sum() + y.sum() + self.eps)


class JaccardLoss(nn.Module):
    def __init__(self, apply_softmax: bool = False, eps: float = 1e-6):
        super().__init__()

        self.apply_softmax = apply_softmax
        self.eps = eps

    def forward(self, x, y, eps=1e-6):
        if self.apply_softmax:
            x = torch.softmax(x, dim=1)

        x = x.view(-1)
        y = y.reshape(-1)

        intersection = (x * y).sum()
        total = (x + y).sum()
        union = total - intersection

        IoU = (intersection + eps) / (union + eps)

        return 1 - IoU


def get_losses(args):
    losses = []
    loss_fn = []
    status_tpl = ["Loss: {:.5f}"]
    if LossType.CCE in args.losses:
        cce_loss = nn.CrossEntropyLoss()

        def cce_loss_fn(logits, targets, num_classes):
            return cce_loss(logits, targets)

        losses.append(cce_loss)
        loss_fn.append(cce_loss_fn)
        status_tpl.append("CCE: {:.5f}")

    if LossType.Jaccard in args.losses:
        jaccard_loss = JaccardLoss(True, 1e-6)

        def jaccard_loss_fn(logits, targets, num_classes):
            tgt_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
            return jaccard_loss(logits, tgt_one_hot)

        losses.append(jaccard_loss)
        loss_fn.append(jaccard_loss_fn)
        status_tpl.append("Jaccard: {:.5f}")

    if LossType.DICE in args.losses:
        dice_loss = DiceCoefficientLoss(True, 1e-6)

        def dice_loss_fn(logits, targets, num_classes):
            tgt_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
            return dice_loss(logits, tgt_one_hot, multiclass=True)

        losses.append(dice_loss)
        loss_fn.append(dice_loss_fn)
        status_tpl.append("DICE: {:.5f}")

    return losses, loss_fn, " | ".join(status_tpl)
