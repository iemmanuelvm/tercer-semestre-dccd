import torch
import torch.nn as nn


def same_padding(kernel_size, dilation=1):
    return ((kernel_size - 1) * dilation) // 2


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, max(1, channels // r), kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(max(1, channels // r), channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.gate(s)
        return x * s


class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, dilation=1, dropout=0.0):
        super().__init__()
        pad = same_padding(k, dilation)
        self.depth = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad,
                               dilation=dilation, groups=in_ch, bias=False)
        self.point = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(
            32, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ResidualTCNBlock(nn.Module):
    def __init__(self, ch, k=7, dilations=(1, 2), dropout=0.05, use_se=True):
        super().__init__()
        layers = []
        in_ch = ch
        for d in dilations:
            layers.append(DepthwiseSeparableConv1D(
                in_ch, ch, k=k, dilation=d, dropout=dropout))
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.se = SqueezeExcite1D(ch) if use_se else nn.Identity()

    def forward(self, x):
        out = self.net(x)
        out = self.se(out)
        return x + out


class BottleneckAttention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=ch, num_heads=num_heads, batch_first=False)
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        x_lbc = x.permute(2, 0, 1)
        y, _ = self.attn(x_lbc, x_lbc, x_lbc, need_weights=False)
        y = self.ln(y)
        y = x_lbc + y
        return y

    def to_conv(self, x_attn):
        return x_attn.permute(1, 2, 0)


class ResUNetTCN(nn.Module):
    def __init__(self, in_ch=1, base=64, depth=3, k=7, dropout=0.05, heads=4):
        super().__init__()
        self.cfg = dict(in_ch=in_ch, base=base, depth=depth,
                        k=k, dropout=dropout, heads=heads)
        self.stem = nn.Conv1d(in_ch, base, kernel_size=3, padding=1)
        enc_blocks, downs = [], []
        ch = base
        for _ in range(depth):
            enc_blocks.append(ResidualTCNBlock(
                ch, k=k, dilations=(1, 2, 4), dropout=dropout))
            downs.append(nn.Conv1d(ch, ch*2, kernel_size=2, stride=2))
            ch *= 2
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downs = nn.ModuleList(downs)
        self.bottleneck = ResidualTCNBlock(
            ch, k=k, dilations=(1, 2, 4, 8), dropout=dropout)
        self.attn = BottleneckAttention(ch, num_heads=heads)
        dec_blocks, ups = [], []
        for _ in range(depth):
            ups.append(nn.ConvTranspose1d(ch, ch//2, kernel_size=2, stride=2))
            ch = ch//2
            dec_blocks.append(ResidualTCNBlock(
                ch, k=k, dilations=(1, 2, 4), dropout=dropout))
        self.ups = nn.ModuleList(ups)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.proj = nn.Conv1d(base, 1, kernel_size=1)

    def _encode_features(self, x):
        skips = []
        h = self.stem(x)
        for blk, down in zip(self.enc_blocks, self.downs):
            h = blk(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        h = self.attn.to_conv(self.attn(h))
        feat = torch.mean(h, dim=-1)
        return h, feat, skips

    def encode(self, x):
        _, feat, _ = self._encode_features(x)
        return feat

    def forward(self, x, return_features=False):
        h, feat, skips = self._encode_features(x)
        for up, blk in zip(self.ups, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                if diff > 0:
                    h = nn.functional.pad(h, (0, diff))
                else:
                    h = h[..., :skip.shape[-1]]
            h = h + skip
            h = blk(h)
        delta = self.proj(h)
        out = x + delta
        if return_features:
            return out, feat
        return out
