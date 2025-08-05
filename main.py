
import numpy as np
import random
import math
import os
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
from data_preparation_runner import prepare_data

[X_train_EOG, y_train_EOG, X_test_EOG, y_test_EOG] = prepare_data(
    combin_num=11, train_per=0.9, noise_type='EMG')


X_train_EOG = torch.FloatTensor(X_train_EOG)
y_train_EOG = torch.FloatTensor(y_train_EOG)
X_test_EOG = torch.FloatTensor(X_test_EOG)
y_test_EOG = torch.FloatTensor(y_test_EOG)

print("X_train_EOG:", X_train_EOG.shape)
print("y_train_EOG:", y_train_EOG.shape)
print("X_test_EOG: ", X_test_EOG.shape)
print("y_test_EOG: ", y_test_EOG.shape)


# ---- Repro ----


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------
# Utilidad: aplanar test [11, 3740, 1, 512] -> [41140, 1, 512]


def flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and x.size(-2) == 1 and x.size(-1) == 512:
        return x.reshape(-1, x.size(-2), x.size(-1))
    return x

# ===== Reemplaza estas variables por tus tensores ya cargados =====
# X_train_EOG: torch.Size([33660, 1, 512])
# y_train_EOG: torch.Size([33660, 1, 512])
# X_test_EOG : torch.Size([11, 3740, 1, 512])
# y_test_EOG : torch.Size([11, 3740, 1, 512])


X_tr = X_train_EOG.float().to(device)
y_tr = y_train_EOG.float().to(device)
X_te = flatten_if_needed(X_test_EOG).float().to(device)
y_te = flatten_if_needed(y_test_EOG).float().to(device)

train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
test_dl = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)

# ---------------------------------------------------------
# Bloques de la CNN (U-Net 1D)


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
        # upsample ×2 manteniendo longitud exacta (512 después de 3 ups)
        self.up = nn.ConvTranspose1d(
            in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # asegurar misma longitud (por si hay desfases por bordes)
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = nn.functional.pad(x, (0, diff))
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class UNet1D(nn.Module):
    """
    Entrada:  [B, 1, 512]
    Salida:   [B, 1, 512]
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
        self.up1 = Up(base*16, base*8)   # concat con enc4 -> canales base*16
        self.up2 = Up(base*8,  base*4)   # concat con enc3 -> canales base*8
        self.up3 = Up(base*4,  base*2)   # concat con enc2 -> canales base*4
        self.up4 = Up(base*2,  base)     # concat con enc1 -> canales base*2
        # Cabeza
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


model = UNet1D(base=64, dropout=0.1).to(device)
print("Parámetros (M):", sum(p.numel() for p in model.parameters())/1e6)

# ---------------------------------------------------------
# Entrenamiento
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5, verbose=True)

epochs = 100
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
best_val, best_state = float('inf'), None

for ep in range(1, epochs+1):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_dl:
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(xb)
            loss = criterion(pred, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(train_dl.dataset)

    # Evaluación
    model.eval()
    te_mse, te_mae = 0.0, 0.0
    with torch.no_grad():
        for xb, yb in test_dl:
            pred = model(xb)
            l = criterion(pred, yb)
            te_mse += l.item() * xb.size(0)
            te_mae += torch.mean(torch.abs(pred - yb)).item() * xb.size(0)
    te_mse /= len(test_dl.dataset)
    te_mae /= len(test_dl.dataset)
    scheduler.step(te_mse)

    print(
        f"Época {ep:02d} | train MSE {tr_loss:.6f} | test MSE {te_mse:.6f} | test MAE {te_mae:.6f}")

    if te_mse < best_val:
        best_val, best_state = te_mse, {
            k: v.detach().cpu() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)
torch.save(model.state_dict(), "cnn1d_unet_emg.pt")
print("Guardado en cnn1d_unet_eog.pt")


# [X_train_EMG, y_train_EMG, X_test_EMG, y_test_EMG] = prepare_data(
#     combin_num=11, train_per=0.9, noise_type='EMG')

# X_train_EMG = torch.FloatTensor(X_train_EMG)
# y_train_EMG = torch.FloatTensor(y_train_EMG)
# X_test_EMG = torch.FloatTensor(X_test_EMG)
# y_test_EMG = torch.FloatTensor(y_test_EMG)


# print("X_train_EMG:", X_train_EMG.shape)
# print("y_train_EMG:", y_train_EMG.shape)
# print("X_test_EMG: ", X_test_EMG.shape)
# print("y_test_EMG: ", y_test_EMG.shape)
