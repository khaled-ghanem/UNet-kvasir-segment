import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive (Conv2d → BatchNorm → ReLU) blocks — the UNet building block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv followed by MaxPool. Returns both the skip connection and the pooled output."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    """Upsample, concatenate skip connection, then DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        # Bilinear upsample avoids the checkerboard artifacts from transposed convolutions
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet for binary segmentation.

    Input:  (B, 3, H, W)
    Output: (B, 1, H, W)  — raw logits, apply sigmoid + threshold at inference
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()

        # Encoder: each block halves spatial dims, doubles channels
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck: deepest representation, no pooling
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder: each block doubles spatial dims, halves channels
        self.dec4 = DecoderBlock(in_ch=1024, skip_ch=512, out_ch=512)
        self.dec3 = DecoderBlock(in_ch=512,  skip_ch=256, out_ch=256)
        self.dec2 = DecoderBlock(in_ch=256,  skip_ch=128, out_ch=128)
        self.dec1 = DecoderBlock(in_ch=128,  skip_ch=64,  out_ch=64)

        # 1×1 conv to project to the number of output classes
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        return self.head(x)
