import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)

    # Flatten spatial dims so intersection/union are computed per sample, then averaged
    probs   = probs.view(probs.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    union        = probs.sum(dim=1) + targets.sum(dim=1)

    # smooth prevents division-by-zero on empty masks and stabilizes gradients early in training
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # BCEWithLogitsLoss combines sigmoid + BCE in one numerically stable operation
    return F.binary_cross_entropy_with_logits(logits, targets)


class BCEDiceLoss(nn.Module):
    """
    BCE stabilizes per-pixel gradients; Dice directly optimizes overlap and
    handles class imbalance. Equal weights work well for polyp segmentation.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * bce_loss(logits, targets) + \
               self.dice_weight * dice_loss(logits, targets)
