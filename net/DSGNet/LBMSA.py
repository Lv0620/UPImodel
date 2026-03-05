import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """轻量通道注意力：Avg/Max + 1x1 -> ReLU -> 1x1 -> Sigmoid"""
    def __init__(self, inp: int, ratio: int = 16):
        super().__init__()
        hidden = max(1, inp // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(inp, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, inp, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        maxv = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        w = self.sigmoid(avg + maxv)
        return x * w


class _MultiScaleStrip1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilations=(1, 2, 3)):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.branches = nn.ModuleList()
        for d in dilations:
            pad = d * (kernel_size - 1) // 2
            self.branches.append(
                nn.Conv1d(channels, channels, kernel_size=kernel_size,
                          padding=pad, dilation=d, groups=channels, bias=False)
            )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sum 多分支
        y = None
        for conv in self.branches:
            out = conv(x)
            y = out if y is None else (y + out)
        y = self.bn(y)
        y = self.act(y)
        return y  # (B, C, L), 值域[0,1]

class _SingleScaleStrip1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, d=1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        pad = d * (kernel_size - 1) // 2
        self.conv1d = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                          padding=pad, dilation=d, groups=channels, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sum 多分支
        y = None
        y = self.conv1d(x)
        y = self.bn(y)
        y = self.act(y)
        return y  # (B, C, L), 值域[0,1]


class _DirectionalWeight(nn.Module):
    def  __init__(self, channels: int, kernel_size: int = 7, dilations=(1, 2, 3)):
        super().__init__()
        self.squeeze = nn.Conv2d(2, 1, kernel_size=1, bias=True)  # 压 2->1
        self.strip = _MultiScaleStrip1D(channels, kernel_size=kernel_size, dilations=dilations)
        #self.strip = _SingleScaleStrip1D(channels, kernel_size=kernel_size, d=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L_ortho, C, L)
        b, l_ortho, c, l = x.size()
        # 沿正交维（dim=1）做 max/avg，得到 (B,1,C,L)
        pooled_max = torch.max(x, dim=1, keepdim=True)[0]
        pooled_avg = torch.mean(x, dim=1, keepdim=True)
        pooled = torch.cat([pooled_max, pooled_avg], dim=1)   # (B,2,C,L)
        s = self.squeeze(pooled)                               # (B,1,C,L)

        # 转成 (B,C,L)，做多尺度 strip-1D 权重
        s = s.view(b, c, l)                                    # (B,C,L)
        w = self.strip(s)                                      # (B,C,L) in [0,1]
        w = w.view(b, 1, c, l)                                 # (B,1,C,L)
        return w


class LBMSA(nn.Module):
    def __init__(self, channels: int, ksize: int = 5, dilations=(1, 2, 3), ca_ratio: int = 16):
        super().__init__()
        self.h_weight = _DirectionalWeight(channels, kernel_size=ksize, dilations=dilations)
        self.w_weight = _DirectionalWeight(channels, kernel_size=ksize, dilations=dilations)


        self.gate_fc = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=channels, bias=False),  # depthwise
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.mix_linear = nn.Linear(2, 2, bias=True)

        self.ca = ChannelAttention(channels, ratio=ca_ratio)


        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def _orientation_mix(self, wh: torch.Tensor, ww: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # wh, ww: (B, C, L)
        gh = self.gate_fc(wh)              # (B,C,L)
        gw = self.gate_fc(ww)              # (B,C,L)
        gh = gh.mean(dim=-1)               # (B,C)
        gw = gw.mean(dim=-1)               # (B,C)
        g = torch.stack([gh, gw], dim=-1)  # (B,C,2)
        logits = self.mix_linear(g)        # (B,C,2)
        lambdas = F.softmax(logits, dim=-1)
        lambda_h = lambdas[..., 0].unsqueeze(-1)  # (B,C,1)
        lambda_w = lambdas[..., 1].unsqueeze(-1)  # (B,C,1)
        return lambda_h, lambda_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.size()

        x_h_in = x.permute(0, 3, 1, 2).contiguous()
        w_h = self.h_weight(x_h_in)                    # (B,1,C,H)
        x_h_out = (x_h_in * w_h).permute(0, 2, 3, 1).contiguous()  # -> (B,C,H,W)

        x_w_in = x.permute(0, 2, 1, 3).contiguous()
        w_w = self.w_weight(x_w_in)                    # (B,1,C,W)
        x_w_out = (x_w_in * w_w).permute(0, 2, 1, 3).contiguous()  # -> (B,C,H,W)

        wh_seq = w_h.view(b, c, h)
        ww_seq = w_w.view(b, c, w)
        lambda_h, lambda_w = self._orientation_mix(wh_seq, ww_seq)  # (B,C,1),(B,C,1)

        x_h_mix = x_h_out * lambda_h.unsqueeze(-1)     # (B,C,H,W)
        x_w_mix = x_w_out * lambda_w.unsqueeze(-1)     # (B,C,H,W)

        x_ca = self.ca(x)

        out = x + self.alpha * (x_h_mix + x_w_mix) + self.beta * x_ca
        return out

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, H, W = 1, 32, 256, 256
    x = torch.randn(B, C, H, W, device=device)

    model = LightCoordChannelEnhance(channels=C, ksize=5, dilations=(1, 2, 3), ca_ratio=16).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_m = total_params
    print(f"Total Parameters: {params_in_m} ")

    y = model(x)

    print(model)
