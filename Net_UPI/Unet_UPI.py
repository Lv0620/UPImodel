"""
-*- coding: utf-8 -*-
@Project: newwork20241004
@File    : Unet.py
@Author  : Yi-ze
@Time    : 2024-11-11 20:56:22
---- 👇 ♻注入☯灵力♻ 👇----
"""
import torch.nn as nn
import torch
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)  # //为整数除法

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetpointrend(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetpointrend, self).__init__()
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes  # 输出类别数
        self.bilinear = bilinear  # 上采样方式

        self.inc = DoubleConv(n_channels, 64)  # 输入层
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)  # 输出层

    def forward(self, x):
        x1 = self.inc(x)  # 一开始输入b-64-256-256
        x2 = self.down1(x1)  # 四层左部分b-128-128-128 *
        x3 = self.down2(x2)   #b-256-64-64
        x4 = self.down3(x3)   #b-512-32-32
        x5 = self.down4(x4)   #b-512-16-16

        up1x = self.up1(x5, x4)  # 四层右部分b-256-32-32
        up2x = self.up2(up1x, x3) #b-128-64-64
        up3x = self.up3(up2x, x2) #b-64-128-128 *
        x = self.up4(up3x, x1)    #b-64-256-256
        logits = self.outc(x)  # 最终输出
        # xout =torch.cat((x2, up3x),dim=1) #197
        # xout =torch.cat((x3, up2x),dim=1) #389

        return {'up2': up2x, 'coarse':  logits}   #默认up2x  参与了up1x和up3x对比

if __name__ == "__main__":
    model = UNetpointrend(3, 5)
    a = torch.zeros([2, 3, 256, 256])
    out = model(a)
    # print(out.size())
    for k, v in out.items():
        print(f"{k}: {v.shape if hasattr(v, 'shape') else type(v)}")