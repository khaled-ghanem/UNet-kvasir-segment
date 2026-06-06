import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_sample(img_dir: str, mask_dir: str):
    """Load one image-mask pair and print basic properties."""
    name = sorted(os.listdir(img_dir))[0]
    img  = Image.open(os.path.join(img_dir, name))
    mask = Image.open(os.path.join(mask_dir, name))
    print(f"Name : {name}")
    print(f"Image: size={img.size}  mode={img.mode}")
    print(f"Mask : size={mask.size}  mode={mask.mode}")
    return img, mask, name


def inspect_mask(mask) -> np.ndarray:
    """Print raw pixel value distribution of a mask before binarization."""
    arr = np.array(mask)
    print(f"Min: {arr.min()}  Max: {arr.max()}")
    print(f"Unique values: {np.unique(arr)}")
    return arr


def binarize(mask) -> np.ndarray:
    """
    Threshold mask to {0, 1}.

    JPEG compression creates edge artifacts (e.g. values 1-7, 248-254) between
    the pure black (0) and pure white (255) regions. Thresholding at 127 recovers
    a clean binary mask.
    """
    arr = np.array(mask) if not isinstance(mask, np.ndarray) else mask
    binary = (arr > 127).astype(np.uint8)
    print(f"Unique values after binarize: {np.unique(binary)}")
    return binary


def plot_comparison(img, mask, binary_mask: np.ndarray) -> None:
    """Show original image, raw mask, and binarized mask side by side."""
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);                             axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray");               axes[1].set_title("Mask (original)")
    axes[2].imshow(binary_mask * 255, cmap="gray");  axes[2].set_title("Mask (binarized)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def polyp_coverage(binary_mask: np.ndarray) -> float:
    """
    Fraction of pixels labeled as polyp (foreground).

    A model that predicts all-background achieves high pixel accuracy but zero
    usefulness on imbalanced data like this. This ratio motivates using Dice loss.
    """
    return float(binary_mask.sum()) / binary_mask.size


if __name__ == "__main__":
    IMG_DIR  = "data/Kvasir-SEG/images"
    MASK_DIR = "data/Kvasir-SEG/masks"

    img, mask, name = load_sample(IMG_DIR, MASK_DIR)
    inspect_mask(mask)
    binary = binarize(mask)
    plot_comparison(img, mask, binary)
    print(f"\nPolyp coverage: {polyp_coverage(binary):.2%}")
