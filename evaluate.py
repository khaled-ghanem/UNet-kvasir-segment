import os

import matplotlib.pyplot as plt
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def load_checkpoint(model, path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint — epoch {checkpoint['epoch']} | "
          f"val Dice: {checkpoint['val_dice']:.4f} | val IoU: {checkpoint['val_iou']:.4f}")
    return model


def predict(model, imgs: torch.Tensor, device: torch.device, threshold: float = 0.5) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(imgs.to(device))
        return (torch.sigmoid(logits) > threshold).float().cpu()


def show_predictions(model, loader, device, n_samples: int = 4, save_dir: str = None) -> None:
    """Show image / ground truth / prediction for n_samples. Optionally save to save_dir."""
    imgs, masks = next(iter(loader))
    preds = predict(model, imgs, device)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    axes[0, 0].set_title("Image",          fontsize=13)
    axes[0, 1].set_title("Ground Truth",   fontsize=13)
    axes[0, 2].set_title("Prediction",     fontsize=13)

    for i in range(n_samples):
        img  = np.clip(imgs[i].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        gt   = masks[i].squeeze().numpy()
        pred = preds[i].squeeze().numpy()

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt,   cmap="gray")
        axes[i, 2].imshow(pred, cmap="gray")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "predictions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()
