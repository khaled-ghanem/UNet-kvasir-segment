import os

import torch
from tqdm import tqdm

from loss import BCEDiceLoss
from metrics import dice_score, iou_score

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR             = 1e-4
EPOCHS         = 50
PATIENCE       = 10
CHECKPOINT_DIR = "checkpoints"


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(loader, desc="  train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="  val  ", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            total_loss += criterion(logits, masks).item()
            total_dice += dice_score(logits, masks)
            total_iou  += iou_score(logits, masks)
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def overfit_one_batch(model, loader, criterion, optimizer, device, n_steps=50):
    """
    Verify that model + loss are wired up correctly before committing to full training.
    Loss should drop to near-zero within ~20 steps on a single batch.
    """
    model.train()
    imgs, masks = next(iter(loader))
    imgs, masks = imgs.to(device), masks.to(device)

    print("Sanity check: overfitting one batch")
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"  step {step:3d} | loss: {loss.item():.4f}")


def train(model, train_loader, val_loader, optimizer, criterion, device,
          epochs=EPOCHS, patience=PATIENCE, checkpoint_dir=CHECKPOINT_DIR):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    best_dice = 0.0
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(1, epochs + 1):
        train_loss              = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        status = ""
        if val_dice > best_dice:
            best_dice = val_dice
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
            }, checkpoint_path)
            status = " ← best"
        else:
            no_improve += 1
            status = f" (no improve {no_improve}/{patience})"
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}{status}")
                break

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train {train_loss:.4f} | "
              f"val {val_loss:.4f} | "
              f"dice {val_dice:.4f} | "
              f"iou {val_iou:.4f}{status}")

    return history
