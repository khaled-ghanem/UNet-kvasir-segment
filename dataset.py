import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 256
SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _build_transforms(split: str) -> A.Compose:
    # Resize first so augmentations run on the smaller 256x256 array, not the raw variable-size image
    resize = A.Resize(IMAGE_SIZE, IMAGE_SIZE)
    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    to_tensor = ToTensorV2()

    if split == "train":
        return A.Compose([
            resize,
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            normalize,
            to_tensor,
        ])
    return A.Compose([resize, normalize, to_tensor])


def split_dataset(
    img_dir: str, mask_dir: str
) -> tuple[list, list, list, list, list, list]:
    filenames = sorted(os.listdir(img_dir))
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(filenames))

    def paths(idxs):
        imgs  = [os.path.join(img_dir,  filenames[i]) for i in idxs]
        masks = [os.path.join(mask_dir, filenames[i]) for i in idxs]
        return imgs, masks

    train_imgs, train_masks = paths(indices[:800])
    val_imgs,   val_masks   = paths(indices[800:900])
    test_imgs,  test_masks  = paths(indices[900:])

    return train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks


class KvasirDataset(Dataset):
    def __init__(self, img_paths: list[str], mask_paths: list[str], split: str) -> None:
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.transform  = _build_transforms(split)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.array(Image.open(self.img_paths[idx]).convert("RGB"))

        # Masks are stored as RGB JPEGs; convert to grayscale to get a single channel
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        # JPEG compression introduces values between 0 and 255 at polyp edges; threshold to {0,1}
        mask = (mask > 127).astype(np.uint8)

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"]                          # float32 [3, H, W]
        mask_tensor  = augmented["mask"].unsqueeze(0).float()      # float32 [1, H, W]

        return image_tensor, mask_tensor


def build_dataloaders(
    img_dir: str,
    mask_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks = split_dataset(
        img_dir, mask_dir
    )

    train_ds = KvasirDataset(train_imgs, train_masks, split="train")
    val_ds   = KvasirDataset(val_imgs,   val_masks,   split="val")
    test_ds  = KvasirDataset(test_imgs,  test_masks,  split="test")

    pin = torch.cuda.is_available()

    # drop_last=True on train prevents a final batch of size 1,
    # which would cause BatchNorm to crash during training
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader
