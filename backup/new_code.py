# CleanEEGNet: EMG removal from EEG (PyTorch)
# Works with tensors shaped [N, 1, 512]

import random
from data_preparation_runner import prepare_data
import math
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from contextlib import nullcontext

# -----------------------------
# Utils
# -----------------------------


def set_seed(seed: int = 123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def snr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SNR in dB for batch of [B, 1, T] tensors."""
    ref = reference.reshape(reference.size(0), -1)
    est = estimate.reshape(estimate.size(0), -1)
    num = (ref**2).sum(dim=1) + eps
    den = ((ref - est)**2).sum(dim=1) + eps
    return 10.0 * torch.log10(num / den)


@torch.no_grad()
def snri(x_noisy: torch.Tensor, y_clean: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """SNR improvement (SNRi): SNR(y,y_hat) - SNR(y,x_noisy). Inputs: [B,1,T]"""
    return snr(y_clean, y_hat) - snr(y_clean, x_noisy)

# -----------------------------
# Losses
# -----------------------------


class MultiResSTFTLoss(nn.Module):
    """Multi-resolution STFT magnitude L1 loss for 1D signals."""

    def __init__(self, ffts=(32, 64, 128), hops=(8, 16, 32), wins=(32, 64, 128)):
        super().__init__()
        assert len(ffts) == len(hops) == len(wins)
        self.ffts = ffts
        self.hops = hops
        self.wins = wins

    def stft_mag(self, x, n_fft, hop, win):
        # x: [B,1,T] -> complex STFT -> magnitude
        x = x.squeeze(1)  # [B, T]
        window = torch.hann_window(win, device=x.device, dtype=x.dtype)
        Z = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, return_complex=True, center=True, pad_mode="reflect"
        )
        mag = torch.abs(Z)  # [B, F, frames]
        return mag

    def forward(self, y_hat, y):
        loss = 0.0
        for n_fft, hop, win in zip(self.ffts, self.hops, self.wins):
            YH = self.stft_mag(y_hat, n_fft, hop, win)
            Y = self.stft_mag(y,     n_fft, hop, win)
            loss = loss + F.l1_loss(YH, Y)
        return loss / len(self.ffts)


def si_snr_loss(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative SI-SNR (to minimize). Inputs [B,1,T]"""
    s = target.squeeze(1)
    s_hat = estimate.squeeze(1)
    s_zm = s - s.mean(dim=1, keepdim=True)
    s_hat_zm = s_hat - s_hat.mean(dim=1, keepdim=True)
    dot = torch.sum(s_hat_zm * s_zm, dim=1, keepdim=True)
    s_energy = torch.sum(s_zm**2, dim=1, keepdim=True) + eps
    s_target = dot / s_energy * s_zm
    e_noise = s_hat_zm - s_target
    ratio = (torch.sum(s_target**2, dim=1) + eps) / \
        (torch.sum(e_noise**2, dim=1) + eps)
    si_snr = 10.0 * torch.log10(ratio + eps)
    return -(si_snr.mean())


class HybridDenoiseLoss(nn.Module):
    """L = lambda_t * L1 + lambda_f * MR-STFT + lambda_sisnr * (-SI-SNR)"""

    def __init__(self, lambda_t=0.5, lambda_f=0.4, lambda_sisnr=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mrstft = MultiResSTFTLoss()
        self.lambda_t = lambda_t
        self.lambda_f = lambda_f
        self.lambda_sisnr = lambda_sisnr

    def forward(self, y_hat, y):
        return (self.lambda_t * self.l1(y_hat, y)
                + self.lambda_f * self.mrstft(y_hat, y)
                + self.lambda_sisnr * si_snr_loss(y_hat, y))

# -----------------------------
# Model blocks
# -----------------------------


class SE1D(nn.Module):
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        assert channels >= r, "SE reduction would make zero channels; lower r or raise channels."
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // r, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(channels // r, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class DWSeparableConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1, padding=None):
        super().__init__()
        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size, padding=padding,
                            dilation=dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1, use_se=True):
        super().__init__()
        assert in_ch % 8 == 0 and out_ch % 8 == 0, "GroupNorm requires channels % 8 == 0 here."
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = DWSeparableConv1D(
            in_ch, out_ch, kernel_size, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = DWSeparableConv1D(out_ch, out_ch, kernel_size, dilation=1)
        self.se = SE1D(out_ch) if use_se else nn.Identity()
        self.skip = nn.Conv1d(
            in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        h = self.se(h)
        return h + self.skip(x)


class MHSA1D(nn.Module):
    """Multi-Head Self-Attention across the temporal axis at the bottleneck."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "embed dim must be divisible by heads"
        self.proj_in = nn.Conv1d(channels, channels, 1, bias=False)
        self.attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):  # x: [B,C,T]
        h = self.proj_in(x)
        h_seq = self.norm(h.transpose(1, 2))  # [B,T,C]
        h_out, _ = self.attn(h_seq, h_seq, h_seq, need_weights=False)
        h_out = h_out.transpose(1, 2)  # [B,C,T]
        return self.proj_out(h_out) + x


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.res = ResBlock1D(in_ch, out_ch, dilation=dilation)
        self.down = nn.Conv1d(
            out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.res(x)
        skip = x
        x = self.down(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res = ResBlock1D(out_ch*2, out_ch, dilation=dilation)

    def forward(self, x, skip):
        x = self.up(x)
        # robustly align temporal length
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (0, diff))
            elif diff < 0:
                x = x[..., :skip.size(-1)]
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        return x


class CleanEEGNet(nn.Module):
    """
    1D ResUNet with SE and MHSA bottleneck.
    Default for input length 512. Handles other multiples-of-32 lengths too.
    """

    def __init__(self, in_channels=1, base_channels=48, levels=5, heads=4):
        super().__init__()
        chs = [base_channels * (2 ** i)
               for i in range(levels)]  # 48,96,192,384,768
        self.stem = nn.Conv1d(in_channels, chs[0], kernel_size=7, padding=3)

        enc, dil = [], 1
        for i in range(levels-1):
            enc.append(EncoderBlock(chs[i], chs[i+1], dilation=dil))
            dil = min(dil*2, 8)
        self.encoder = nn.ModuleList(enc)

        self.bottleneck = nn.Sequential(
            ResBlock1D(chs[-1], chs[-1], dilation=4),
            MHSA1D(chs[-1], num_heads=heads),
            ResBlock1D(chs[-1], chs[-1], dilation=1),
        )

        dec = []
        for i in range(levels-2, -1, -1):
            in_ch, out_ch = chs[i+1], chs[i]
            dec.append(DecoderBlock(in_ch, out_ch, dilation=1))
        self.decoder = nn.ModuleList(dec)

        self.head = nn.Sequential(
            nn.Conv1d(chs[0], chs[0]//2, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(chs[0]//2, in_channels, kernel_size=1)
        )

    def forward(self, x):
        # --- stem ---
        x_stem = self.stem(x)              # [B, 48, T]
        # keep stem for the final decoder concat
        skips = [x_stem]

        # --- encoder ---
        x = x_stem
        for i, block in enumerate(self.encoder):
            # s channels: 96, 192, 384, 768 (across depth)
            x, s = block(x)
            # keep all but the deepest skip (we don't use the 768-ch skip)
            if i < len(self.encoder) - 1:
                skips.append(s)           # now skips = [48, 96, 192, 384]

        # --- bottleneck ---
        x = self.bottleneck(x)

        # --- decoder ---
        for block in self.decoder:
            s = skips.pop()               # pops 384, 192, 96, 48
            x = block(x, s)

        noise_hat = self.head(x)
        return noise_hat  # wrapper will subtract from input


class CleanEEGWrapper(nn.Module):
    """Wrapper to do residual prediction: y_hat = x - f(x)"""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x):
        noise_hat = self.core(x)
        return x - noise_hat

# -----------------------------
# Data
# -----------------------------


def make_loaders(X_train, y_train, X_test, y_test,
                 batch_size=256, num_workers=0, normalize_per_window=True):
    """
    Builds dataloaders from provided tensors shaped [N,1,512].
    Optionally per-window z-score normalize (recommended for biosignals).
    """
    def _norm(a):
        if not normalize_per_window:
            return a
        mean = a.mean(dim=-1, keepdim=True)
        std = a.std(dim=-1, keepdim=True) + 1e-8
        return (a - mean) / std

    Xtr = _norm(X_train.clone().float())
    Ytr = _norm(y_train.clone().float())
    Xte = _norm(X_test.clone().float())
    Yte = _norm(y_test.clone().float())

    train_ds = TensorDataset(Xtr, Ytr)
    test_ds = TensorDataset(Xte, Yte)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    return train_dl, test_dl

# -----------------------------
# Metrics (MSE, RMSE, RRMSE, CC)
# -----------------------------


@torch.no_grad()
def _accumulate_metrics(y_hat: torch.Tensor, y: torch.Tensor,
                        sums: dict) -> None:
    """
    Accumulate metrics over a batch for epoch-level aggregation.
    - MSE   = SSE / N
    - RMSE  = sqrt(MSE)
    - RRMSE = sqrt( SSE / sum(y^2) )
    - CC    = mean Pearson correlation per-sample (across T)
    """
    # Flatten per sample
    yh = y_hat.reshape(y_hat.size(0), -1)
    yt = y.reshape(y.size(0), -1)

    # SSE and target power
    e = yh - yt
    sse = torch.sum(e * e).item()
    ssy = torch.sum(yt * yt).item()

    # per-sample Pearson correlation
    yh_c = yh - yh.mean(dim=1, keepdim=True)
    yt_c = yt - yt.mean(dim=1, keepdim=True)
    num = torch.sum(yh_c * yt_c, dim=1)
    den = torch.sqrt(torch.sum(yh_c**2, dim=1) *
                     torch.sum(yt_c**2, dim=1)) + 1e-8
    cc = (num / den).clamp(min=-1.0, max=1.0)  # [B]
    cc_sum = torch.sum(cc).item()

    # Accumulate
    sums["sse"] += sse
    sums["ssy"] += ssy
    sums["n"] += yt.numel()
    sums["cc_sum"] += cc_sum
    sums["cc_count"] += yt.size(0)


def _finalize_metrics(sums: dict) -> Tuple[float, float, float, float]:
    mse = sums["sse"] / max(1, sums["n"])
    rmse = math.sqrt(mse)
    rrmse = math.sqrt(sums["sse"] / max(1e-12, sums["ssy"]))
    cc = sums["cc_sum"] / max(1, sums["cc_count"])
    return mse, rmse, rrmse, cc


def _zero_metric_sums() -> dict:
    return {"sse": 0.0, "ssy": 0.0, "n": 0, "cc_sum": 0.0, "cc_count": 0}

# -----------------------------
# Training
# -----------------------------


class EarlyStopper:
    def __init__(self, patience=8, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.should_stop = False

    def step(self, metric):
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train(
    X_train, y_train, X_test, y_test,
    epochs=30, batch_size=256, lr=2e-3, weight_decay=1e-4,
    device=None, save_path="cleaneegnet.pt"
):
    set_seed(42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, test_dl = make_loaders(
        X_train, y_train, X_test, y_test, batch_size=batch_size)

    core = CleanEEGNet(base_channels=48, levels=5, heads=4)
    model = CleanEEGWrapper(core).to(device)
    loss_fn = HybridDenoiseLoss(lambda_t=0.5, lambda_f=0.4, lambda_sisnr=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr*0.1)

    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    stopper = EarlyStopper(patience=6, min_delta=1e-5)

    best_val_mse = float("inf")
    for epoch in range(1, epochs+1):
        t0 = time.time()

        # ------------------ Train ------------------
        model.train()
        run_loss = 0.0
        n_loss = 0
        train_sums = _zero_metric_sums()

        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with (autocast_ctx if use_amp else nullcontext()):
                y_hat = model(xb)
                loss = loss_fn(y_hat, yb)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            run_loss += loss.item() * xb.size(0)
            n_loss += xb.size(0)

            # accumulate metrics on-the-fly
            _accumulate_metrics(y_hat.detach(), yb, train_sums)

        scheduler.step()
        train_loss = run_loss / max(1, n_loss)
        train_mse, train_rmse, train_rrmse, train_cc = _finalize_metrics(
            train_sums)

        # ------------------ Validation ------------------
        model.eval()
        val_loss_sum, n_val_loss = 0.0, 0
        val_sums = _zero_metric_sums()

        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                y_hat = model(xb)
                val_loss_sum += loss_fn(y_hat, yb).item() * xb.size(0)
                n_val_loss += xb.size(0)
                _accumulate_metrics(y_hat, yb, val_sums)

        val_loss = val_loss_sum / max(1, n_val_loss)
        val_mse, val_rmse, val_rrmse, val_cc = _finalize_metrics(val_sums)

        dt = time.time() - t0
        print(
            f"[{epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  |  "
            f"TRAIN: MSE={train_mse:.6e} RMSE={train_rmse:.6f} RRMSE={train_rrmse:.6f} CC={train_cc:.4f}  |  "
            f"VAL:   MSE={val_mse:.6e} RMSE={val_rmse:.6f} RRMSE={val_rrmse:.6f} CC={val_cc:.4f}  |  "
            f"time={dt:.1f}s"
        )

        # Save best by validation MSE (your primary metric)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({"model": model.state_dict()}, save_path)

        # if stopper.step(val_mse):
        #     print("Early stopping triggered.")
        #     break

    print(f"Best val MSE: {best_val_mse:.6e}  (weights saved to {save_path})")
    return model

# -----------------------------
# Inference helper
# -----------------------------


@torch.no_grad()
def denoise(model: nn.Module, x: torch.Tensor, device=None, batch_size=1024) -> torch.Tensor:
    """Denoise in batches. x: [N,1,T]"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    outs = []
    for i in range(0, x.size(0), batch_size):
        xb = x[i:i+batch_size].to(device)
        yb = model(xb)
        outs.append(yb.cpu())
    return torch.cat(outs, dim=0)

# -----------------------------
# Example run
# -----------------------------


if __name__ == "__main__":
    # Reproducibilidad y backend
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # Datos
    X_train, y_train, X_test, y_test = prepare_data(
        combin_num_train=11,         # heavy augmentation for train
        combin_num_test=1,           # light permute for test
        train_per=0.98,              # tiny test split
        noise_type="EMG",
        num_test_snrs=3,             # fewer SNR points (e.g., -5, 0, +5 dB)
        max_test_segments=8000,      # cap total test windows after flatten
        flatten_test=True,           # returns [N,1,512] for test
        normalize="paired",          # do paired z-score here
        seed=42
    )

    # Flatten any [B, T, 1, 512] into [N, 1, 512]
    X_train = torch.as_tensor(X_train, dtype=torch.float32).reshape(-1, 1, 512)
    y_train = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1, 512)
    X_test = torch.as_tensor(X_test,  dtype=torch.float32).reshape(-1, 1, 512)
    y_test = torch.as_tensor(y_test,  dtype=torch.float32).reshape(-1, 1, 512)

    print(f"X_train {X_train.shape}")
    print(f"y_train {y_train.shape}")
    print(f"X_test  {X_test.shape}")
    print(f"y_test  {y_test.shape}")

    model = train(X_train, y_train, X_test, y_test,
                  epochs=50, batch_size=256, lr=2e-3, weight_decay=1e-4,
                  device=None, save_path="cleaneegnet.pt")

    # Example: denoise test set and compute final metrics
    with torch.no_grad():
        y_hat_test = denoise(model, X_test)
        sums = _zero_metric_sums()
        _accumulate_metrics(y_hat_test, y_test, sums)
        mse, rmse, rrmse, cc = _finalize_metrics(sums)
        print(
            f"[FINAL TEST] MSE={mse:.6e} RMSE={rmse:.6f} RRMSE={rrmse:.6f} CC={cc:.4f}")
