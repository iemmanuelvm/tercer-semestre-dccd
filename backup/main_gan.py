# parallel_gan_transformer_denoise.py
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Data helpers (match your shapes)
# -------------------------


class EEGPairs(Dataset):
    """
    Expects tensors shaped:
      train: X [N, 1, 512], y [N, 1, 512]
      test:  X [B, T, 1, 512], y [B, T, 1, 512] -> flattened to [B*T, 1, 512]
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.dim() in (3, 4), f"Unexpected X dim {X.shape}"
        assert y.dim() in (3, 4), f"Unexpected y dim {y.shape}"
        if X.dim() == 4:
            b, t, c, l = X.shape
            X = X.reshape(b * t, c, l)
            y = y.reshape(b * t, c, l)
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------
# Building blocks
# -------------------------

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, p=None, norm=True):
        super().__init__()
        if p is None:
            p = k // 2
        layers = [nn.Conv1d(in_ch, out_ch, k, s, p), nn.GELU()]
        if norm:
            layers.insert(1, nn.BatchNorm1d(out_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock1D(in_ch, out_ch, k=7, s=2),  # stride 2 downsample
            ConvBlock1D(out_ch, out_ch, k=5, s=1),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            ConvBlock1D(out_ch * 2, out_ch, k=5, s=1),
            ConvBlock1D(out_ch, out_ch, k=5, s=1),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:  # pad if rounding mismatch
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(
            0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(
            1), persistent=False)  # [L, 1, D]

    def forward(self, x):
        # x: [L, B, D]
        L = x.shape[0]
        return x + self.pe[:L]


class TransformerBranch(nn.Module):
    """
    Processes sequence [B, 1, 512] -> embeds to D and applies TransformerEncoder.
    """

    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1, seq_len=512):
        super().__init__()
        self.proj = nn.Conv1d(1, d_model, kernel_size=1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = PositionalEncoding1D(d_model, max_len=seq_len)
        # compress for fusion
        self.head = nn.Conv1d(d_model, 64, kernel_size=1)

    def forward(self, x):
        # x: [B, 1, L]
        z = self.proj(x)                         # [B, D, L]
        z_t = z.transpose(1, 2).transpose(0, 1)  # [L, B, D]
        z_t = self.pos(z_t)
        z_t = self.encoder(z_t)                  # [L, B, D]
        z = z_t.transpose(0, 1).transpose(1, 2)  # [B, D, L]
        z = self.head(z)                         # [B, 64, L]
        return z


class UNet1D(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.inc = nn.Sequential(
            ConvBlock1D(in_ch, base_ch, k=7),
            ConvBlock1D(base_ch, base_ch, k=5),
        )
        self.down1 = Down(base_ch, base_ch * 2)     # 64 -> 128
        self.down2 = Down(base_ch * 2, base_ch * 4)  # 128 -> 256
        self.down3 = Down(base_ch * 4, base_ch * 4)  # 256 -> 256

        self.bottleneck = nn.Sequential(
            ConvBlock1D(base_ch * 4, base_ch * 4, k=3),
            ConvBlock1D(base_ch * 4, base_ch * 4, k=3),
        )

        self.up3 = Up(base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch)
        self.outc = nn.Conv1d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bottleneck(x4)
        u3 = self.up3(xb, x3)
        u2 = self.up2(u3, x2)
        u1 = self.up1(u2, x1)
        out = self.outc(u1)
        return out, (x1, x2, x3)


class ParallelFusionGenerator(nn.Module):
    """
    Parallel-guided: UNet branch (local detail) + Transformer branch (global context).
    Learned gates produce fused noise estimate; output is residual: clean = noisy - noise_hat.
    """

    def __init__(self, seq_len=512, base_ch=64, t_d_model=128, t_layers=4, t_heads=8):
        super().__init__()
        self.unet = UNet1D(in_ch=1, base_ch=base_ch)
        self.transformer = TransformerBranch(
            d_model=t_d_model, nhead=t_heads, num_layers=t_layers,
            dim_feedforward=t_d_model*2, seq_len=seq_len
        )
        self.proj_unet = nn.Conv1d(base_ch, 64, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv1d(64*2, 64, kernel_size=1),
            nn.GELU(),
            # 2 gates -> softmax along channel
            nn.Conv1d(64, 2, kernel_size=1),
        )
        self.head = nn.Sequential(
            ConvBlock1D(64, 64, k=5),
            nn.Conv1d(64, 1, kernel_size=1)
        )

    def forward(self, noisy):
        unet_out, (x1, _, _) = self.unet(noisy)  # unet_out: [B,1,L]
        unet_feat = self.proj_unet(x1)           # [B,64,L]
        trans_feat = self.transformer(noisy)     # [B,64,L]

        fused_cat = torch.cat([unet_feat, trans_feat], dim=1)  # [B,128,L]
        gates = self.gate(fused_cat)                            # [B,2,L]
        gates = F.softmax(gates, dim=1)
        g_unet, g_trans = gates[:, 0:1, :], gates[:, 1:2, :]

        fused = g_unet * unet_feat + g_trans * trans_feat       # [B,64,L]
        # residual boost
        noise_hat = self.head(fused) + unet_out
        clean_hat = noisy - noise_hat
        return clean_hat, noise_hat


class PatchDiscriminator1D(nn.Module):
    """1-D PatchGAN-style discriminator for (signal) realism."""

    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()

        def block(ci, co, k=15, s=2, p=None):
            if p is None:
                p = k // 2
            return nn.Sequential(
                nn.Conv1d(ci, co, k, s, p),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(co, co, 5, 1, 2),
                nn.BatchNorm1d(co),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 15, 2, 7),
            nn.LeakyReLU(0.2, inplace=True),
            block(base_ch, base_ch*2),
            block(base_ch*2, base_ch*4),
            nn.Conv1d(base_ch*4, 1, kernel_size=3,
                      stride=1, padding=1),  # patch logits
        )

    def forward(self, x):
        return self.net(x)  # [B, 1, ~L/8]


# -------------------------
# Losses
# -------------------------

class SpectralConvergenceLoss(nn.Module):
    """Optional: compare magnitude spectra for stability against phase jitter."""

    def __init__(self, n_fft=256, hop_length=64, win_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.window = torch.hann_window(win_length)

    def forward(self, pred, target):
        # pred, target: [B, 1, L]
        device = pred.device
        if self.window.device != device:
            self.window = self.window.to(device)
        S_pred = torch.stft(
            pred.squeeze(1), n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            window=self.window, return_complex=True, center=True
        ).abs()
        S_tgt = torch.stft(
            target.squeeze(1), n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            window=self.window, return_complex=True, center=True
        ).abs()
        num = torch.norm(S_pred - S_tgt, p='fro')
        den = torch.norm(S_tgt, p='fro') + 1e-8
        return num / den


# -------------------------
# Training utilities
# -------------------------

@dataclass
class TrainConfig:
    seq_len: int = 512
    batch_size: int = 256
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    lambda_l1: float = 100.0
    lambda_spec: float = 1.0
    epochs: int = 20
    num_workers: int = 2
    amp: bool = True


def make_loaders(X_train, y_train, X_test, y_test, cfg: TrainConfig):
    ds_train = EEGPairs(X_train, y_train)
    ds_val = EEGPairs(X_test, y_test)
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


# -------------------------
# Full training loop with TRAIN/VAL MSE per epoch
# -------------------------

def train_model(X_train_EMG, y_train_EMG, X_test_EMG, y_test_EMG,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    cfg = TrainConfig()
    train_loader, val_loader = make_loaders(
        X_train_EMG, y_train_EMG, X_test_EMG, y_test_EMG, cfg)

    G = ParallelFusionGenerator(seq_len=cfg.seq_len).to(device)
    D = PatchDiscriminator1D().to(device)

    opt_g = torch.optim.AdamW(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(D.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))

    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    spec_loss = SpectralConvergenceLoss()
    mse_loss_sum = nn.MSELoss(reduction="sum")  # accumulate across samples

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    for epoch in range(1, cfg.epochs + 1):
        G.train()
        D.train()
        g_running, d_running = 0.0, 0.0

        # Accumulate train MSE over all samples
        train_mse_sum = 0.0
        train_count = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            # -------- Train D --------
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                clean_hat_detached, _ = G(noisy.detach())
                real_logits = D(clean)
                fake_logits = D(clean_hat_detached.detach())
                d_real = adv_loss(real_logits, torch.ones_like(real_logits))
                d_fake = adv_loss(fake_logits, torch.zeros_like(fake_logits))
                d_loss = (d_real + d_fake) * 0.5

            opt_d.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)

            # -------- Train G --------
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                clean_hat, _ = G(noisy)
                fake_logits = D(clean_hat)
                g_adv = adv_loss(fake_logits, torch.ones_like(fake_logits))
                g_l1 = l1_loss(clean_hat, clean)
                g_spec = spec_loss(clean_hat, clean)
                g_loss = g_adv + cfg.lambda_l1 * g_l1 + cfg.lambda_spec * g_spec

                # accumulate train MSE (sum) and count
                train_mse_sum += mse_loss_sum(clean_hat, clean).item()
                train_count += clean.numel()

            opt_g.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()

            g_running += g_loss.item()
            d_running += d_loss.item()

        train_mse = train_mse_sum / train_count

        # -------- Validation --------
        G.eval()
        val_mse_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                pred, _ = G(noisy)
                val_mse_sum += mse_loss_sum(pred, clean).item()
                val_count += clean.numel()
        val_mse = val_mse_sum / val_count

        print(f"Epoch {epoch:02d} | "
              f"G {g_running/len(train_loader):.3f} | "
              f"D {d_running/len(train_loader):.3f} | "
              f"Train MSE {train_mse:.8f} | Val MSE {val_mse:.8f}")

    return G, D


# -------------------------
# Inference helper
# -------------------------

@torch.no_grad()
def denoise_segments(G: nn.Module, X: torch.Tensor, device=None, batch_size=1024):
    """
    X can be [N, 1, 512] or [B, T, 1, 512].
    Returns the same shape as X with denoised signals.
    """
    device = device or next(G.parameters()).device
    orig_shape = X.shape
    if X.dim() == 4:
        b, t, c, l = X.shape
        Xf = X.reshape(b * t, c, l)
    else:
        Xf = X
    outs = []
    for i in range(0, Xf.shape[0], batch_size):
        noisy = Xf[i:i+batch_size].to(device)
        clean_hat, _ = G(noisy)
        outs.append(clean_hat.cpu())
    Yf = torch.cat(outs, dim=0)
    if len(orig_shape) == 4:
        Yf = Yf.reshape(orig_shape)
    return Yf


# -------------------------
# Script entrypoint
# -------------------------

if __name__ == "__main__":
    # Replace with your actual data loading
    from data_preparation_runner import prepare_data

    X_train_EMG, y_train_EMG, X_test_EMG, y_test_EMG = prepare_data(
        combin_num=11, train_per=0.9, noise_type='EMG'
    )

    X_train_EMG = torch.FloatTensor(X_train_EMG)  # [N, 1, 512]
    y_train_EMG = torch.FloatTensor(y_train_EMG)  # [N, 1, 512]
    X_test_EMG = torch.FloatTensor(X_test_EMG)   # [11, 6160, 1, 512]
    y_test_EMG = torch.FloatTensor(y_test_EMG)   # [11, 6160, 1, 512]

    print("X_train_EMG:", X_train_EMG.shape)
    print("y_train_EMG:", y_train_EMG.shape)
    print("X_test_EMG: ", X_test_EMG.shape)
    print("y_test_EMG: ", y_test_EMG.shape)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    G, D = train_model(X_train_EMG, y_train_EMG, X_test_EMG,
                       y_test_EMG, device=device)

    # Quick shape check on denoising output
    denoised = denoise_segments(G, X_test_EMG, device=device)
    print("denoised:", denoised.shape)
