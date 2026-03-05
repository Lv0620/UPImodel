import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBNReLU_1x1(nn.Module):
    """Conv-BN-ReLU_1x1 用于调整通道数"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, is_act=True):
        super(ConvBNReLU_1x1, self).__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if is_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv_1x1(x)


class _DWConv3x1_1x3(nn.Module):
    """可选 BatchNorm 和 ReLU 的深度可分离卷积, kernel_size=3"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_bn=True, activation=True):
        super(_DWConv3x1_1x3, self).__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                      groups=in_channels, bias=not use_bn)
        ]

        if use_bn:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation:
            layers.append(nn.ReLU6(inplace=True))  # 适用于量化
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 1, bias=not use_bn)
        ])
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.ReLU6(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class _DWConv5x1_1x5(nn.Module):
    """5x1和1x5深度可分离卷积，用于深层网络扩展感受野"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(_DWConv5x1_1x5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, (5, 1), stride=stride, padding=(2, 0), groups=in_channels,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, (1, 5), stride=stride, padding=(0, 2), groups=in_channels,
                               bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)  # 逐点卷积，调整通道数
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _GhostModule(nn.Module):
    """Ghost Module from GhostNet"""

    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(_GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = out_channels // ratio
        new_channels = init_channels * (ratio - 1)

        # 主特征
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding=0, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU6(inplace=True) if relu else nn.Identity()
            # inplace=True 就地修改x，不会重新创建，节省内存，nn.Identity()表示占位层，输入什么输出什么
        )

        # Ghost 特征
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU6(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        cheap = self.cheap_operation(primary)
        return torch.cat([primary, cheap], dim=1)


class GDFB(nn.Module):
    """GDFB with GhostModule and DWConv"""

    def __init__(self, in_channels, inter_channels, out_channels):
        super(GDFB, self).__init__()
        self.conv1 = ConvBNReLU_1x1(in_channels, inter_channels, kernel_size=1)  # 降低计算量
        self.conv2 = _DWConv3x1_1x3(inter_channels, inter_channels, kernel_size=3,
                                    stride=1)  # 3x3的卷积提取特征，并将特征图尺寸缩小为原来的一半，进行了下采样
        self.conv3 = _GhostModule(inter_channels, out_channels)  # 恢复通道数量，1x1的卷积
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)  # 降低计算量
        out = self.conv2(out)  # 卷积
        out = self.conv3(out)  # 恢复通道
        out = self.act(out + identity)
        return out

class Down_GDFB(nn.Module):
    """Down_GDFB with GhostModule and DWConv"""

    def __init__(self, in_channels, inter_channels, out_channels):
        super(Down_GDFB, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.down = ConvBNReLU_1x1(in_channels, out_channels, kernel_size=1, is_act=False)
        self.conv1 = ConvBNReLU_1x1(in_channels, inter_channels, kernel_size=1)  # 降低计算量
        self.conv2 = _DWConv3x1_1x3(inter_channels, inter_channels, kernel_size=3,
                                    stride=2)  # 3x3的卷积提取特征，并将特征图尺寸缩小为原来的一半，进行了下采样
        self.conv3 = _GhostModule(inter_channels, out_channels)  # 恢复通道数量，1x1的卷积
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        identity = x
        identity = self.maxpool(identity)
        identity = self.down(identity)
        out = self.conv1(x)  # 降低计算量
        out = self.conv2(out)  # 卷积
        out = self.conv3(out)  # 恢复通道
        out = self.act(out + identity)
        return out



class Broad_GDFB(nn.Module):
    """Bottleneck Block with GhostModule and DWConv"""

    def __init__(self, in_channels, inter_channels, out_channels):
        super(Broad_GDFB, self).__init__()
        self.conv1 = ConvBNReLU_1x1(in_channels, inter_channels, kernel_size=1)  # 降低计算量
        self.conv2 = _DWConv5x1_1x5(inter_channels, inter_channels)  # 用于深层，保持较大的感受野，特征图的大小保持不变
        self.conv3 = _GhostModule(inter_channels, out_channels)  # 恢复通道数量，1x1的卷积
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)  # 降低计算量
        out = self.conv2(out)  # 卷积
        out = self.conv3(out)  # 恢复通道
        out = self.act(out + identity)
        return out


class DoubleGhostBlock(nn.Module):  # 加深网络，扩展感受野
    '''带残差结构的双GhostModule'''

    def __init__(self, in_channels, inter_channels, out_channels):
        super(DoubleGhostBlock, self).__init__()
        self.Ghost1 = _GhostModule(in_channels, inter_channels, stride=1)  # 将通道扩张
        # #  # 深度可分离卷积
        # self.depthwise = nn.Sequential(
        #     nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels, bias=False),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU6(inplace=True)
        # )
        self.Ghost2 = _GhostModule(inter_channels, out_channels, stride=1, relu=False)  # 将通道恢复到原来的尺寸
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        out_x = self.Ghost1(x)
        out_x = self.Ghost2(out_x)
        out_x = self.act(out_x + x)  # 残差连接
        return out_x

class HierarchicalAdaptiveFusion(nn.Module):
    """
    轻量层级自适应上采样融合模块，用于两个不同尺度的特征图融合 + 注意力增强
    """

    def __init__(self, in_channels, out_channels):  # in_channel 深层特征图的通道数，out_channel 浅层特征图的通道数
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.low_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.align_low = ConvBNReLU_1x1(in_channels, out_channels)

        # 自适应融合门控：根据上下层特征生成权重α
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        # 通道注意力增强模块（可换成 SE / CoordAtt 等）
        self.channel_attention = ECA(out_channels)

        # 输出卷积
        self.out_conv = _DWConv3x1_1x3(out_channels, out_channels)

    def forward(self, x_low, x_high):
        # x_low: 低分辨率特征 [B, C, H/2, W/2]
        # x_high: 高分辨率特征 [B, C, H, W]
        x_low_up = self.low_upsample(x_low)  # 上采样
        if self.in_channels != self.out_channels:
            x_low_up = self.align_low(x_low_up)  # 通道对齐
        # 拼接生成融合权重 α ∈ [0,1]
        fused = torch.cat([x_low_up, x_high], dim=1)  # [B, 2C, H, W]
        alpha = self.fuse_gate(fused)  # [B, C, H, W]

        # 自适应加权融合
        fusion = alpha * x_high + (1 - alpha) * x_low_up

        # 通道注意力增强
        fusion = self.channel_attention(fusion)

        # 输出卷积进一步处理
        out = self.out_conv(fusion)
        return out


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # 动态计算 kernel size
        k_size = self.get_kernel_size(channels, gamma, b)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def get_kernel_size(self, c, gamma, b):
        k = int(abs((math.log2(c) / gamma) + b))
        # 保证为奇数
        if k % 2 == 0:
            k += 1
        return k

    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B, 1, C] → Conv1D
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)

class SegHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(SegHead, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.inter_channels = in_channels // 2
        self.conv1x1 = ConvBNReLU_1x1(in_channels, self.inter_channels)
        self.conv1 = _DWConv3x1_1x3(self.inter_channels, self.inter_channels)
        self.conv_out = nn.Conv2d(self.inter_channels, n_classes, kernel_size=3, padding=1)

    def forward(self, x, size):
        out = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 在这里是向上四倍上采样到原图大小
        out = self.conv1x1(out)
        out = self.conv1(out)
        out = self.conv_out(out)
        return out


class UpFuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpFuse, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvBNReLU_1x1(in_channels, out_channels)
        self.fuse_bn = nn.BatchNorm2d(out_channels)
        self.fuse_act = nn.ReLU6()
        self.fuse_atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # x1: 通常是高层特征
        # x2: 通常是低层特征
        if self.in_channels != self.out_channels:
            x1 = self.conv1(x1)
        x1 = F.interpolate(x1, x2.size()[2:], mode='bilinear', align_corners=True)
        fuse_feature = torch.add(x1, x2)
        fuse_feature = self.fuse_bn(fuse_feature)
        fuse_feature = self.fuse_act(fuse_feature)
        fuse_feature_att = self.fuse_atten(fuse_feature)
        return fuse_feature * fuse_feature_att



