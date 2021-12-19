import torch
import torch.nn.functional as F


def logits_to_onehot(logits, num_classes):
    sig_logits = torch.sigmoid(logits)
    return F.one_hot(sig_logits.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
