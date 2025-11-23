# -*- coding: utf-8 -*-
"""
Denoising EMG en EEG con U-Net 1D (ventanas de 512)
Correcciones:
- base=48 (más capacidad)
- AdamW lr=3e-3, weight_decay=3e-4
- Warmup lineal 3 épocas + CosineAnnealingWarmRestarts per-iter
- Early stopping patience=40
- GroupNorm (BN->GN)
- Decoder: Upsample + Conv1d (evita artefactos)
- Pérdida Charbonnier + MR-STFT (w_spec=0.05)
- EMA (AveragedModel) para evaluar
- Normalización z-score consistente (según y limpio)
- Métricas en escala original (des-normalizando)
- Aumentos ligeros en train (ganancia/shift)
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.swa_utils import AveragedModel
from typing import Tuple

from neural_network_runner import Transformer1D  # opcional, no usado aquí
from data_preparation_runner import prepare_data

# ---------------------------
# Reproducibilidad y backend
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

# ---------------------------
# Datos
X_train, y_train, X_test, y_test = prepare_data(
    combin_num=11,
    train_per=0.9,
    noise_type="EMG",
)

# Tensores [N,1,512]
X_train = torch.as_tensor(X_train, dtype=torch.float32).reshape(-1, 1, 512)
y_train = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1, 512)
X_test = torch.as_tensor(X_test,  dtype=torch.float32).reshape(-1, 1, 512)
y_test = torch.as_tensor(y_test,  dtype=torch.float32).reshape(-1, 1, 512)

print(f"X_train {X_train.shape}")
print(f"y_train {y_train.shape}")
print(f"X_test  {X_test.shape}")
print(f"y_test  {y_test.shape}")

# ---------------------------
# Normalización z-score consistente (usar stats del objetivo limpio)
with torch.no_grad():
    mu = y_train.mean().item()
    sigma = y_train.std().item() + 1e-8


def norm(t): return (t - mu) / sigma
def denorm(t): return t * sigma + mu


X_train = norm(X_train)
y_train = norm(y_train)
X_test = norm(X_test)
y_test = norm(y_test)

# ---------------------------
# Aumentos ligeros (opcionales) para train


def augment(xb: torch.Tensor) -> torch.Tensor:
    # xb: [B,1,512] normalizado
    if not torch.is_grad_enabled():
        return xb
    B = xb.size(0)
    # Ganancia aleatoria ±5%
    gain = (1.0 + 0.05 * (2*torch.rand(B, 1, 1, device=xb.device)-1.0))
    xb = xb * gain
    # Pequeño shift circular [-8..8] muestras
    max_shift = 8
    shifts = torch.randint(-max_shift, max_shift+1, (B,), device=xb.device)
    if max_shift > 0:
        for i, s in enumerate(shifts):
            if s != 0:
                xb[i] = torch.roll(xb[i], shifts=int(s.item()), dims=-1)
    return xb

# ---------------------------
# Modelo


def GN(c):  # GroupNorm helper
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ResDilatedBlock(nn.Module):
    def __init__(self, ch, kernel=15, dilation=1):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.gn1 = GN(ch)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.gn2 = GN(ch)
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        out = self.prelu1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.prelu2(out + x)


class Encoder(nn.Module):
    def __init__(self, cin, cout, kernel=8, stride=2):
        super().__init__()
        pad = (kernel - stride) // 2
        self.down = nn.Conv1d(cin, cout, kernel_size=kernel,
                              stride=stride, padding=pad)
        self.gn = GN(cout)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.gn(self.down(x)))


class Decoder(nn.Module):
    # Upsample + Conv1d para evitar artefactos de ConvTranspose
    def __init__(self, cin, cout, scale=2, kernel=5):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale,
                              mode='linear', align_corners=False)
        self.conv = nn.Conv1d(cin, cout, kernel_size=kernel, padding=kernel//2)
        self.gn = GN(cout)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.gn(self.conv(self.up(x))))


class DenoiseUNet1D(nn.Module):
    def __init__(self, base=48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, kernel_size=7, padding=3),
            GN(base),
            nn.PReLU(),
        )
        self.enc1 = Encoder(base,   base*2)
        self.rb1 = ResDilatedBlock(base*2, dilation=1)
        self.enc2 = Encoder(base*2, base*4)
        self.rb2 = ResDilatedBlock(base*4, dilation=2)
        self.enc3 = Encoder(base*4, base*8)
        self.rb3 = ResDilatedBlock(base*8, dilation=4)
        self.enc4 = Encoder(base*8, base*16)
        self.rb4 = ResDilatedBlock(base*16, dilation=8)

        self.bott = ResDilatedBlock(base*16, dilation=16)

        self.dec4 = Decoder(base*16, base*8)
        self.rb4u = ResDilatedBlock(base*16, dilation=4)
        self.dec3 = Decoder(base*16, base*4)
        self.rb3u = ResDilatedBlock(base*8, dilation=2)
        self.dec2 = Decoder(base*8,  base*2)
        self.rb2u = ResDilatedBlock(base*4, dilation=1)
        self.dec1 = Decoder(base*4,  base)
        self.rb1u = ResDilatedBlock(base*2, dilation=1)

        self.head = nn.Conv1d(base*2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        x0 = self.stem(x)
        e1 = self.rb1(self.enc1(x0))
        e2 = self.rb2(self.enc2(e1))
        e3 = self.rb3(self.enc3(e2))
        e4 = self.rb4(self.enc4(e3))

        b = self.bott(e4)

        d4 = self.dec4(b)
        d4 = self.rb4u(torch.cat([d4, e3], dim=1))
        d3 = self.dec3(d4)
        d3 = self.rb3u(torch.cat([d3, e2], dim=1))
        d2 = self.dec2(d3)
        d2 = self.rb2u(torch.cat([d2, e1], dim=1))
        d1 = self.dec1(d2)
        d1 = self.rb1u(torch.cat([d1, x0], dim=1))

        pred_noise = self.head(d1)
        denoised = x - pred_noise
        return denoised, pred_noise

# ---------------------------
# Pérdida (Charbonnier + MR-STFT con peso bajo)


def stft_mag(x, n_fft, hop, win):
    X = torch.stft(
        x.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win,
        window=torch.hann_window(win, device=x.device),
        return_complex=True
    )
    return torch.abs(X)


def mrstft_loss(pred, target):
    cfgs = [(64, 16, 64), (128, 32, 128), (256, 64, 256)]
    loss = 0.0
    for n_fft, hop, win in cfgs:
        P = stft_mag(pred, n_fft, hop, win)
        T = stft_mag(target, n_fft, hop, win)
        sc = torch.norm(T - P, p='fro') / (torch.norm(T, p='fro') + 1e-8)
        l1 = F.l1_loss(P, T)
        loss += sc + l1
    return loss / len(cfgs)


class Charbonnier(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.sqrt((x - y).pow(2) + self.eps**2).mean()


class DenoiseLoss(nn.Module):
    def __init__(self, w_time=1.0, w_spec=0.05):
        super().__init__()
        self.l_time = Charbonnier()
        self.w_time = w_time
        self.w_spec = w_spec

    def forward(self, y_hat, y):
        return self.w_time * self.l_time(y_hat, y) + self.w_spec * mrstft_loss(y_hat, y)

# ---------------------------
# Métricas (en escala original)


@torch.no_grad()
def init_metric_state():
    return {
        "numel": 0.0,
        "sum_e2": 0.0,   # para MSE/RMSE/RRMSE
        "sum_y2": 0.0,   # para RRMSE
        # para CC global (Pearson):
        "N": 0.0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2_cc": 0.0,
        "sum_xy": 0.0,
    }


@torch.no_grad()
def update_metric_state(state, y_hat, y):
    # y_hat, y: [B,1,512] EN ESCALA ORIGINAL
    e = (y_hat - y).float()
    x = y_hat.float()
    t = y.float()
    state["sum_e2"] += e.pow(2).sum().item()
    state["sum_y2"] += t.pow(2).sum().item()
    n = e.numel()
    state["numel"] += n
    state["N"] += n
    state["sum_x"] += x.sum().item()
    state["sum_y"] += t.sum().item()
    state["sum_x2"] += x.pow(2).sum().item()
    state["sum_y2_cc"] += t.pow(2).sum().item()
    state["sum_xy"] += (x * t).sum().item()


@torch.no_grad()
def finalize_metrics(state):
    mse = state["sum_e2"] / max(state["numel"], 1.0)
    rmse = math.sqrt(max(mse, 0.0))
    rrmse = math.sqrt(state["sum_e2"] / max(state["sum_y2"], 1e-12))
    N = state["N"]
    sx, sy = state["sum_x"], state["sum_y"]
    sxx, syy, sxy = state["sum_x2"], state["sum_y2_cc"], state["sum_xy"]
    num = N * sxy - sx * sy
    den = math.sqrt(max(N * sxx - sx * sx, 1e-12)) * \
        math.sqrt(max(N * syy - sy * sy, 1e-12))
    cc = num / den if den > 0 else 0.0
    return mse, rmse, rrmse, cc

# ---------------------------
# OLA (overlap-add) para señales largas (opcional)


@torch.no_grad()
def predict_ola(model, x_long: torch.Tensor, win=512, hop=256, device='cpu'):
    """
    x_long: [1,1,L] normalizado
    Devuelve y_hat en escala original [1,1,L]
    """
    model.eval()
    L = x_long.shape[-1]
    if L <= win:
        y_hat, _ = model(x_long.to(device))
        return denorm(y_hat.cpu())
    # padding para cubrir totalmente
    n_hops = max(1, (L - win + hop) // hop)
    total = n_hops * hop + win
    pad = total - L
    x_pad = F.pad(x_long, (0, pad))
    out = torch.zeros_like(x_pad)
    cnt = torch.zeros_like(x_pad)
    for start in range(0, x_pad.shape[-1] - win + 1, hop):
        seg = x_pad[..., start:start+win].to(device)
        y_hat, _ = model(seg)
        out[..., start:start+win] += y_hat.cpu()
        cnt[..., start:start+win] += 1.0
    out = out / cnt.clamp_min(1.0)
    out = out[..., :L]
    return denorm(out)


# ---------------------------
# Entrenamiento + validación
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenoiseUNet1D(base=48).to(device)
ema_model = AveragedModel(model)  # EMA de pesos

criterion = DenoiseLoss(w_time=1.0, w_spec=0.05)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=3e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, T_0=20, T_mult=2, eta_min=1e-5
)

# Split train/val a partir del training original (no tocamos test)
val_ratio = 0.1
n_total = X_train.shape[0]
n_val = int(n_total * val_ratio)
n_trn = n_total - n_val
train_ds_full = TensorDataset(X_train, y_train)
train_ds, val_ds = random_split(
    train_ds_full, [n_trn, n_val], generator=torch.Generator().manual_seed(seed))
test_ds = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=256,
                          shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds,   batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds,  batch_size=256, shuffle=False)

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Early stopping
best_val_mse = float('inf')
patience = 40
pat_count = 0
best_state_dict = None

epochs = 300
warmup_epochs = 3
global_step = 0

for epoch in range(1, epochs+1):
    # ---------- Train ----------
    model.train()
    train_state = init_metric_state()
    total_loss = 0.0
    steps_per_epoch = max(1, len(train_loader))

    for batch_idx, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        xb = augment(xb)
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            y_hat, _ = model(xb)
            loss = criterion(y_hat, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        # EMA update después del step
        ema_model.update_parameters(model)

        # Scheduler per-iter con warmup 3 primeras épocas
        global_step += 1
        progress = epoch + (batch_idx + 1) / steps_per_epoch
        if epoch <= warmup_epochs:
            warm_factor = ((epoch - 1) * steps_per_epoch +
                           (batch_idx + 1)) / (warmup_epochs * steps_per_epoch)
            warm_factor = max(1e-3, min(1.0, warm_factor))  # evita cero
            for pg in opt.param_groups:
                base_lr = 3e-3
                pg['lr'] = base_lr * warm_factor
        else:
            sched.step(progress)

        total_loss += loss.item()

        # Métricas en escala original
        y_hat_den = denorm(y_hat.detach().cpu())
        yb_den = denorm(yb.detach().cpu())
        update_metric_state(train_state, y_hat_den, yb_den)

    tr_mse, tr_rmse, tr_rrmse, tr_cc = finalize_metrics(train_state)

    # ---------- Validación (con EMA) ----------
    ema_model.eval()
    val_state = init_metric_state()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_hat, _ = ema_model(xb)
            y_hat_den = denorm(y_hat.cpu())
            yb_den = denorm(yb.cpu())
            update_metric_state(val_state, y_hat_den, yb_den)
    val_mse, val_rmse, val_rrmse, val_cc = finalize_metrics(val_state)

    print(
        f"Epoch {epoch:03d} | "
        f"Train MSE {tr_mse:.6e} RMSE {tr_rmse:.6e} RRMSE {tr_rrmse:.6e} CC {tr_cc:.4f} || "
        f"Val   MSE {val_mse:.6e} RMSE {val_rmse:.6e} RRMSE {val_rrmse:.6e} CC {val_cc:.4f}"
    )

    # Early stopping
    if val_mse < best_val_mse - 1e-9:
        best_val_mse = val_mse
        pat_count = 0
        best_state_dict = {k: v.cpu().clone()
                           # guardamos EMA
                           for k, v in ema_model.state_dict().items()}
    else:
        pat_count += 1
        if pat_count >= patience:
            print(
                f"Early stopping en epoch {epoch} (mejor Val MSE={best_val_mse:.6e}).")
            break

# Restaurar mejor EMA y evaluar en test
if best_state_dict is not None:
    ema_model.load_state_dict(best_state_dict)

# ---------- Test (EMA) ----------
ema_model.eval()
test_state = init_metric_state()
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        y_hat, _ = ema_model(xb)
        y_hat_den = denorm(y_hat.cpu())
        yb_den = denorm(yb.cpu())
        update_metric_state(test_state, y_hat_den, yb_den)
te_mse, te_rmse, te_rrmse, te_cc = finalize_metrics(test_state)
print(
    f"TEST | MSE {te_mse:.6e} RMSE {te_rmse:.6e} RRMSE {te_rrmse:.6e} CC {te_cc:.4f}"
)

# (Opcional) guardar modelo EMA
torch.save(ema_model.state_dict(), "denoise_unet1d_emg_ema.pt")
print("Modelo (EMA) guardado en denoise_unet1d_emg_ema.pt")

# ---------------------------
# Ejemplo de uso OLA para una señal larga (opcional):
# x_long = norm(torch.randn(1,1,4096))  # normaliza con las mismas stats (mu,sigma)
# y_long_hat = predict_ola(ema_model, x_long, win=512, hop=256, device=device)
