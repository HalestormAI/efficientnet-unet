import torch
import torch.nn as nn


class DiceCoefficientLoss(nn.Module):
    def __init__(self, apply_softmax: bool = False, eps: int = 1e-6):
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