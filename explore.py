"""
Step 1: Data exploration for Kvasir-SEG dataset.

Run this script to inspect image/mask properties, visualize the binarization
fix for JPEG compression artifacts, and measure polyp class imbalance.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMG_DIR  = "data/Kvasir-SEG/images"
MASK_DIR = "data/Kvasir-SEG/masks"

# --- Basic inspection ---
name = os.listdir(IMG_DIR)[0]
img  = Image.open(f"{IMG_DIR}/{name}")
mask = Image.open(f"{MASK_DIR}/{name}")

print("Image name :", name)
print("Image size :", img.size)
print("Image mode :", img.mode)
print("Mask size  :", mask.size)
print("Mask mode  :", mask.mode)

# --- Raw mask values ---
mask_arr = np.array(mask)
print("\nMin value    :", mask_arr.min())
print("Max value    :", mask_arr.max())
print("Unique values:", np.unique(mask_arr))

# --- Binarization ---
# JPEG compression introduces edge artifacts (values 1-7, 248-254) between the
# pure black (0) and pure white (255) regions. Threshold at 127 to recover a
# clean binary mask.
binary_mask = (mask_arr > 127).astype(np.uint8)
print("\nUnique values after binarize:", np.unique(binary_mask))

# --- Visualization ---
binary_mask_display = (binary_mask * 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img)
axes[0].set_title("Image")
axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Mask (original)")
axes[2].imshow(binary_mask_display, cmap="gray")
axes[2].set_title("Mask (binarized)")
plt.tight_layout()
plt.show()

# --- Class imbalance ---
# If the polyp covers only ~5% of pixels, a model predicting all-background
# gets 95% pixel accuracy but is completely useless.
# This is why we use Dice loss — it directly measures overlap, not accuracy.
polyp_ratio = binary_mask.sum() / binary_mask.size
print(f"\nPolyp coverage: {polyp_ratio:.2%}")
