

'''
DSGNet: A Lightweight Network Integrating Depthwise Separable and Ghost Convolutions for Real-Time Surface Defect Segmentation
https://doi.org/10.1111/exsy.70187   Digital Object Identifier (DOI)
'''
from dsgnet_parts import *
from LBMSA import LBMSA


class GD_Encoder(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=3, block_channels=[3, 32, 64, 128, 128, 256]):
        super(GD_Encoder, self).__init__()
        self.in_channels = in_channels
        # Stage 1: 得到1/2特征图, 输入通道数32，输出通道数64
        self.stage1 = nn.Sequential(
            Down_GDFB(block_channels[0], block_channels[1] // 4, block_channels[1]),
            GDFB(block_channels[1], block_channels[1] // 4, block_channels[1])
        )
        # Stage 2: 得到1/4特征图，输入通道数64，输出通道数128
        self.stage2 = nn.Sequential(
            Down_GDFB(block_channels[1], block_channels[2] // 4, block_channels[2]),
            GDFB(block_channels[2], block_channels[2] // 4, block_channels[2])
        )
        # Stage 3: 得到1/8特征图, 输入通道数128，输出通道数128
        self.stage3 = nn.Sequential(
            Down_GDFB(block_channels[2], block_channels[3] // 4, block_channels[3]),
            GDFB(block_channels[3], block_channels[3] // 4, block_channels[3]),
            Broad_GDFB(block_channels[3], block_channels[3] // 4, block_channels[3])
        )
        # Stage 4: 得到1/16特征图, 输入通道数128，输出通道数128
        self.stage4 = nn.Sequential(
            Down_GDFB(block_channels[3], block_channels[4] // 4, block_channels[4]),
            GDFB(block_channels[4], block_channels[4] // 4, block_channels[4]),
            Broad_GDFB(block_channels[4], block_channels[4] // 4, block_channels[4])
        )
        # Stage 5: 得到1/32特征图, 输入通道数128，输出通道数256
        self.stage5 = nn.Sequential(
            Down_GDFB(block_channels[4], block_channels[5] // 4, block_channels[5]),  # 下采样
            Broad_GDFB(block_channels[5], block_channels[5] // 4, block_channels[5]),
            DoubleGhostBlock(block_channels[5], block_channels[5] * 2, block_channels[5]),  # 使用Ghost模块进行通道的扩展
        )

    def forward(self, x):
        # print("x.size():", x.size())
        x2 = self.stage1(x)
        # x2 = self.edgeEnhance1(x2)
        # print("x2.size():", x2.size())
        x4 = self.stage2(x2)
        # x4 = self.edgeEnhance2(x4)
        # print("x4.size():", x4.size())
        x8 = self.stage3(x4)
        # print("x8.size():", x8.size())
        x16 = self.stage4(x8)
        # print("x16.size():", x16.size())
        x32 = self.stage5(x16)
        # print("x32.size():", x32.size())
        return x2, x4, x8, x16, x32


class DSGNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=3, block_channels=[3, 32, 64, 128, 128, 256], is_enhance=True):
        super(DSGNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.is_enhance = is_enhance

        self.encoder = GD_Encoder(in_channels, block_channels)

        if self.is_enhance:
            self.featureEnhance6 = LBMSA(block_channels[1], ksize=5, ca_ratio=4)

        self.up_fuse4 = HierarchicalAdaptiveFusion(block_channels[5], block_channels[4])
        self.up_fuse3 = HierarchicalAdaptiveFusion(block_channels[4], block_channels[3])
        self.up_fuse2 = HierarchicalAdaptiveFusion(block_channels[3], block_channels[2])
        self.up_fuse1 = HierarchicalAdaptiveFusion(block_channels[2], block_channels[1])
        self.seg_head = SegHead(block_channels[1], n_classes)

    def forward(self, x):
        x_size = x.size()[2:]
        x2, x4, x8, x16, x32 = self.encoder(x)
        feat1, feat2, feat3, feat4, feat5 = x2, x4, x8, x16, x32
        fuse_feture4 = self.up_fuse4(feat5, feat4)  # 1/16
        fuse_feture3 = self.up_fuse3(fuse_feture4, feat3)  # 1/8   b-128-32-32
        fuse_feture2 = self.up_fuse2(fuse_feture3, feat2)  # 1/4  b-64-64-64
        fuse_feture1 = self.up_fuse1(fuse_feture2, feat1)

        fuse_feture1 = self.featureEnhance6(fuse_feture1)

        out_feat = self.seg_head(fuse_feture1, x_size)
        if self.training:
            return out_feat
        return out_feat


if __name__ == '__main__':
    # from torchinfo import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.randn(8, 3, 256, 256).to(device)
    model = DSGNet(in_channels=3, n_classes=5).to(device=device)
    # summary(
    #     model,
    #     input_size=(1, 3, 256, 256),  # batch=8
    #     col_names=("input_size", "output_size", "num_params"),
    #     depth=3,
    #     device="cuda"
    # )

    model.eval()
    out_feature = model(img)
    print(out_feature.shape)
