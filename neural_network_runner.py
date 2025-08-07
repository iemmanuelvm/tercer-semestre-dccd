
import torch
import torch.nn as nn


# ---------------------------------------------------------
# CNN Blocks (U-Net 1D)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=p),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, k, padding=p),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x): return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # upsample ×2 keeping exact length (512 after 3 ups)
        self.up = nn.ConvTranspose1d(
            in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # ensure the same length (in case there are offsets due to edges)
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = nn.functional.pad(x, (0, diff))
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class UNet1D(nn.Module):
    """
    Input: [B, 1, 512]
    Output: [B, 1, 512]
    """

    def __init__(self, base=64, dropout=0.1):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, base, dropout=dropout)      # 512
        self.enc2 = Down(base, base*2)                       # 256
        self.enc3 = Down(base*2, base*4)                     # 128
        self.enc4 = Down(base*4, base*8)                     # 64
        # Bottleneck
        self.bott = ConvBlock(base*8, base*16, dropout=dropout)  # 64
        # Decoder
        self.up1 = Up(base*16, base*8)   # concat with enc4 -> base channels*16
        self.up2 = Up(base*8,  base*4)   # concat with enc3 -> base channels*8
        self.up3 = Up(base*4,  base*2)   # concat with enc2 -> base channels*4
        self.up4 = Up(base*2,  base)     # concat with enc1 -> base channels*2
        # Head
        self.out = nn.Sequential(
            nn.Conv1d(base, base, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base, 1, kernel_size=1)
        )

    def forward(self, x):
        s1 = self.enc1(x)     # [B, base, 512]
        s2 = self.enc2(s1)    # [B, 2b , 256]
        s3 = self.enc3(s2)    # [B, 4b , 128]
        s4 = self.enc4(s3)    # [B, 8b , 64]
        b = self.bott(s4)    # [B,16b , 64]
        u1 = self.up1(b,  s4)  # [B, 8b , 128 -> block -> 8b , 64]
        u2 = self.up2(u1, s3)  # [B, 4b , 128]
        u3 = self.up3(u2, s2)  # [B, 2b , 256]
        u4 = self.up4(u3, s1)  # [B, 1b , 512]
        y = self.out(u4)     # [B, 1  , 512]
        return y
