# -*- coding: utf-8 -*-
"""
Denoising EMG en EEG con U-Net 1D (ventanas de 512)
Métricas por época: MSE, RMSE, RRMSE, CC en train y test
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from neural_network_runner import Transformer1D  # opcional, no usado aquí
from data_preparation_runner import prepare_data

# ---------------------------
# Reproducibilidad y backend
seed = 42
random.seed(seed)
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

X_train = torch.as_tensor(X_train, dtype=torch.float32).reshape(-1, 1, 512)
y_train = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1, 512)
X_test = torch.as_tensor(X_test,  dtype=torch.float32).reshape(-1, 1, 512)
y_test = torch.as_tensor(y_test,  dtype=torch.float32).reshape(-1, 1, 512)

print(f"X_train {X_train.shape}")
print(f"y_train {y_train.shape}")
print(f"X_test  {X_test.shape}")
print(f"y_test  {y_test.shape}")

# ---------------------------
# Modelo


class ResDilatedBlock(nn.Module):
    def __init__(self, ch, kernel=15, dilation=1):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.prelu2(out + x)


class Encoder(nn.Module):
    def __init__(self, cin, cout, kernel=8, stride=2):
        super().__init__()
        pad = (kernel - stride) // 2
        self.down = nn.Conv1d(cin, cout, kernel_size=kernel,
                              stride=stride, padding=pad)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.down(x)))


class Decoder(nn.Module):
    def __init__(self, cin, cout, kernel=8, stride=2):
        super().__init__()
        pad = (kernel - stride) // 2
        self.up = nn.ConvTranspose1d(
            cin, cout, kernel_size=kernel, stride=stride, padding=pad, output_padding=0)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.up(x)))


class DenoiseUNet1D(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, kernel_size=7, padding=3),
            nn.BatchNorm1d(base),
            nn.PReLU(),
        )
        self.enc1 = Encoder(base, base*2)
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
        self.dec2 = Decoder(base*8, base*2)
        self.rb2u = ResDilatedBlock(base*4, dilation=1)
        self.dec1 = Decoder(base*4, base)
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
# Pérdida (MSE + MR-STFT opcional)


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


class DenoiseLoss(nn.Module):
    def __init__(self, w_time=1.0, w_spec=0.5):
        super().__init__()
        self.w_time = w_time
        self.w_spec = w_spec

    def forward(self, y_hat, y):
        return self.w_time * F.mse_loss(y_hat, y) + self.w_spec * mrstft_loss(y_hat, y)

# ---------------------------
# Métricas (acumulación global exacta)


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
    # y_hat, y: [B,1,512]
    e = (y_hat - y).float()
    x = y_hat.float()
    t = y.float()
    # básicos
    state["sum_e2"] += e.pow(2).sum().item()
    state["sum_y2"] += t.pow(2).sum().item()
    n = e.numel()
    state["numel"] += n
    # para CC (sobre todos los elementos)
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
    # RRMSE respecto al RMS del objetivo (||e|| / ||y||)
    rrmse = math.sqrt(state["sum_e2"] / max(state["sum_y2"], 1e-12))
    # CC de Pearson global
    N = state["N"]
    sx, sy = state["sum_x"], state["sum_y"]
    sxx, syy, sxy = state["sum_x2"], state["sum_y2_cc"], state["sum_xy"]
    num = N * sxy - sx * sy
    den = math.sqrt(max(N * sxx - sx * sx, 1e-12)) * \
        math.sqrt(max(N * syy - sy * sy, 1e-12))
    cc = num / den if den > 0 else 0.0
    return mse, rmse, rrmse, cc


# ---------------------------
# Entrenamiento + validación con métricas por época
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenoiseUNet1D(base=32).to(device)
criterion = DenoiseLoss(w_time=1.0, w_spec=0.5)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test,  y_test)
train_loader = DataLoader(train_ds, batch_size=256,
                          shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds,  batch_size=256, shuffle=False)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

epochs = 300
for epoch in range(1, epochs+1):
    # ---------- Train ----------
    model.train()
    train_state = init_metric_state()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            y_hat, _ = model(xb)
            loss = criterion(y_hat, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item()
        # métricas sobre pred actual
        update_metric_state(train_state, y_hat, yb)
    sched.step()
    tr_mse, tr_rmse, tr_rrmse, tr_cc = finalize_metrics(train_state)

    # ---------- Test ----------
    model.eval()
    test_state = init_metric_state()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_hat, _ = model(xb)
            update_metric_state(test_state, y_hat, yb)
    te_mse, te_rmse, te_rrmse, te_cc = finalize_metrics(test_state)

    print(
        f"Epoch {epoch:02d} | "
        f"Train MSE {tr_mse:.6e} RMSE {tr_rmse:.6e} RRMSE {tr_rrmse:.6e} CC {tr_cc:.4f} || "
        f"Test MSE {te_mse:.6e} RMSE {te_rmse:.6e} RRMSE {te_rrmse:.6e} CC {te_cc:.4f}"
    )

# (Opcional) guardar el modelo
torch.save(model.state_dict(), "denoise_unet1d_emg.pt")
print("Modelo guardado en denoise_unet1d_emg.pt")
