import torch


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()

    preds   = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    union        = preds.sum(dim=1) + targets.sum(dim=1)

    # If both pred and target are empty, the model perfectly predicted "no polyp" → score = 1.0
    score = torch.where(union == 0, torch.ones_like(intersection), 2.0 * intersection / union)
    return score.mean().item()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()

    preds   = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    union        = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    score = torch.where(union == 0, torch.ones_like(intersection), intersection / union)
    return score.mean().item()
