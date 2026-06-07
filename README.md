# UNet — Kvasir-SEG Polyp Segmentation

Binary segmentation of colorectal polyps from colonoscopy images, built from scratch with PyTorch.

---

## Pipeline

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `explore.py` | Load a sample image/mask, inspect JPEG artifacts, binarize mask, measure class imbalance (8.91% polyp coverage) |
| 2 | `dataset.py` | `KvasirDataset` — 800/100/100 train/val/test split (seed 42), albumentations transforms (resize 256×256, ImageNet normalize, train augmentations) |
| 3 | `model.py` | UNet — 4-block encoder (64→512 ch), 1024-ch bottleneck, skip-connection decoder, 1×1 head → logits (B, 1, 256, 256) |
| 4 | `loss.py` | `BCEDiceLoss` — 0.5 × BCE + 0.5 × Dice. BCE stabilizes gradients; Dice directly optimizes overlap and handles class imbalance |
| 5 | `metrics.py` | `dice_score`, `iou_score` — computed on binarized predictions (threshold 0.5), edge case: both empty → 1.0 |
| 6 | `train.py` | Epoch loop with tqdm, saves best checkpoint by val Dice, early stopping (patience=10) |
| 7 | `evaluate.py` | Load best checkpoint, evaluate on test set, save prediction overlays |

---

## Dataset

[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) — 1000 colonoscopy images + 1000 binary masks.

```
data/
└── Kvasir-SEG/
    ├── images/   # 1000 JPEG images (variable sizes)
    └── masks/    # 1000 JPEG masks (binarize at threshold 127)
```

Download and extract into `data/` before running.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run

Open `Kvasir-SEG.ipynb` and run all cells in order, or run each module directly:

```bash
python explore.py        # Step 1: data exploration
python train.py          # Steps 2-6: full training pipeline  (edit paths inside)
```

Checkpoints are saved to `checkpoints/best_model.pth`.
Prediction overlays are saved to `outputs/predictions.png`.

---

## Results

| Metric | Test Score |
|--------|-----------|
| Dice   | —         |
| IoU    | —         |

*To be updated after training.*
