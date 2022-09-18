# Simple U-Net
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class PeriodicConv2d(nn.Module):
    def __init__(self, nf_in: int, nf_out: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, **kwargs) -> None:
        super().__init__()
        self.pad_lat = (kernel_size[0] - 1) // 2
        self.pad_lon = (kernel_size[1] - 1) // 2  # ( for ks in kernel_size)
        self.conv = nn.Conv2d(nf_in, nf_out, kernel_size=kernel_size, stride=stride, padding=0, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: input tensor, shape: (bs, c, h, w)"""
        # E-W: periodic padding
        x = F.pad(x, pad=(self.pad_lon, self.pad_lon, 0, 0), mode="circular")
        # zero padding in the latitudinal direction
        x = F.pad(x, pad=(0, 0, self.pad_lat, self.pad_lat), mode="constant", value=0.0)
        # convolution + activation
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            PeriodicConv2d(in_channels, mid_channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            PeriodicConv2d(mid_channels, out_channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = PeriodicConv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, bilinear=False):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.bilinear = bilinear

        self.inc = DoubleConv(num_inputs, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_outputs)

        # self.inc = DoubleConv(num_inputs, 32)
        # self.down1 = Down(32, 64)
        # self.down2 = Down(64, 128)
        # self.down3 = Down(128, 256)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(256, 512 // factor)
        # self.up1 = Up(512, 256 // factor, bilinear)
        # self.up2 = Up(256, 128 // factor, bilinear)
        # self.up3 = Up(128, 64 // factor, bilinear)
        # self.up4 = Up(64, 32, bilinear)
        # self.outc = OutConv(32, num_outputs)

    def initialize_weights(self) -> None:
        """Module weight initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, PeriodicConv2d)):
                nn.init.kaiming_normal_(m.weight, a=self._LRU_ALPHA)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


if __name__ == "__main__":
    bs, num_in, num_out, h, w = 8, 4, 2, 181, 360
    x_in = torch.rand(bs, num_in, h, w)
    x_out = UNet(4, 2, bilinear=True)(x_in)
    assert x_out.shape == (bs, num_out, h, w)
