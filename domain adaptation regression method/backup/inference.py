# ========= GRAFICAR DE 512 EN 512 MUESTRAS (segmento por segmento) =========
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


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


# ---- Usa TU misma definición de la red ----
# class ConvBlock(...), class Down(...), class Up(...), class UNet1D(...)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Helper de forma ---


def is_4d_batches(x: torch.Tensor) -> bool:
    return x.dim() == 4 and x.size(-2) == 1 and x.size(-1) == 512


def to_2d_segments(x: torch.Tensor) -> torch.Tensor:
    """[B1, B2, 1, 512] -> [B1*B2, 1, 512]  ó  [N,1,512] -> igual"""
    return x.reshape(-1, 1, 512) if is_4d_batches(x) else x


def index_segment(x: torch.Tensor, combo: int, seg: int) -> torch.Tensor:
    """
    Devuelve un tensor [1,1,512] del índice pedido.
    - Si x es [B1,B2,1,512]: usa [combo, seg]
    - Si x es [N,1,512]: el índice lineal es combo*B2 + seg (si quieres usarlo así)
    """
    if is_4d_batches(x):
        xi = x[combo, seg]                 # [1,512]
        return xi.unsqueeze(0)             # [1,1,512]
    else:
        # x ya está [N,1,512] -> asume 'combo' es 0 y usa 'seg' como índice lineal
        return x[seg:seg+1]                # [1,1,512]


# --- Carga de pesos (mismo modelo/hiperparámetros que en entrenamiento) ---
model = UNet1D(base=64, dropout=0.1).to(device)
state = torch.load("best_unet1d_cc_rrmse_emg.pth", map_location=device)
model.load_state_dict(state)
model.eval()

use_amp = torch.cuda.is_available()

# --- Utilidades de graficado por segmento ---


@torch.no_grad()
def infer_one(x_1x1x512: torch.Tensor) -> torch.Tensor:
    """x: [1,1,512] -> y_hat: [1,1,512]"""
    with torch.amp.autocast('cuda', enabled=use_amp):
        yhat = model(x_1x1x512.to(device))
    return yhat.cpu()


def plot_segment(x_noisy_1x1x512: torch.Tensor, y_clean_1x1x512: torch.Tensor, y_hat_1x1x512: torch.Tensor,
                 title_prefix=""):
    xn = x_noisy_1x1x512.squeeze().numpy()   # [512]
    yc = y_clean_1x1x512.squeeze().numpy()   # [512]
    yh = y_hat_1x1x512.squeeze().numpy()     # [512]
    mse = float(np.mean((yh - yc) ** 2))
    mae = float(np.mean(np.abs(yh - yc)))

    plt.figure()
    plt.plot(xn, label="Noisy (X)")
    plt.plot(yc, label="Clean (y)")
    plt.plot(yh, label="Denoised (ŷ)")
    plt.title(f"{title_prefix}  MSE={mse:.6f}  MAE={mae:.6f}")
    plt.xlabel("Muestras (512)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_range(X_test_EOG: torch.Tensor, y_test_EOG: torch.Tensor,
               combo: int = 0, seg_ini: int = 0, seg_fin: int = 10,
               pause: float = 0.0, guardar_png: bool = False, carpeta_out: str = "plots_seg"):
    """
    Recorre y grafica los segmentos [seg_ini, seg_fin) de una combinación.
    - combo: índice de combinación (0..B1-1) si tus tensores son [B1,B2,1,512]
    - seg_ini/seg_fin: rango de segmentos (0..B2)
    - pause: pausa (s) entre gráficas; 0 para bloquear hasta cerrar figura
    - guardar_png: si True, guarda cada figura en carpeta_out
    """
    import os
    if guardar_png:
        os.makedirs(carpeta_out, exist_ok=True)

    for s in range(seg_ini, seg_fin):
        x = index_segment(X_test_EOG, combo, s)      # [1,1,512]
        y = index_segment(y_test_EOG, combo, s)      # [1,1,512]
        yhat = infer_one(x)

        title = f"Combo {combo} | Segmento {s}"
        xn = x.squeeze().numpy()
        yc = y.squeeze().numpy()
        yh = yhat.squeeze().numpy()
        mse = float(np.mean((yh - yc) ** 2))
        mae = float(np.mean(np.abs(yh - yc)))

        plt.figure()
        plt.plot(xn, label="Noisy (X)")
        plt.plot(yc, label="Clean (y)")
        plt.plot(yh, label="Denoised (ŷ)")
        plt.title(f"{title} | MSE={mse:.6f} MAE={mae:.6f}")
        plt.xlabel("Muestras (512)")
        plt.ylabel("Amplitud")
        plt.grid(True)
        plt.legend()

        if guardar_png:
            fn = os.path.join(carpeta_out, f"combo{combo:02d}_seg{s:05d}.png")
            plt.savefig(fn, bbox_inches="tight", dpi=120)
        plt.show(block=(pause == 0.0))
        if pause > 0:
            plt.pause(pause)
        plt.close()

# --- (Opcional) Reconstruir señal continua de una combinación y graficar por tramos de 512 ---


def concat_combo(x_combo_2d: torch.Tensor) -> np.ndarray:
    """
    x_combo_2d: [B2, 1, 512]  -> vector 1D de longitud B2*512
    """
    return x_combo_2d.reshape(-1, 512).flatten().cpu().numpy()


@torch.no_grad()
def infer_combo_continuo(X_test_EOG: torch.Tensor, combo: int = 0):
    """
    Devuelve tres vectores 1D (noisy, clean, denoised) concatenando todos los segmentos
    de una combinación.
    """
    assert is_4d_batches(
        X_test_EOG), "Para continuo, se espera forma [B1,B2,1,512]."
    B1, B2, _, _ = X_test_EOG.shape
    xs = []
    ys = []
    yhs = []
    for s in range(B2):
        x = X_test_EOG[combo, s:s+1]  # [1,1,512]
        with torch.amp.autocast('cuda', enabled=use_amp):
            yh = model(x.to(device)).cpu()
        xs.append(x)
        yhs.append(yh)
    X_concat = concat_combo(torch.cat(xs, dim=0))      # [B2*512]
    Yhat_concat = concat_combo(torch.cat(yhs, dim=0))   # [B2*512]
    return X_concat, Yhat_concat


def plot_continuo_ventanas(X_test_EOG: torch.Tensor, y_test_EOG: torch.Tensor, combo: int = 0,
                           ventana: int = 512, desde_seg: int = 0, hasta_seg: int = 20):
    """
    Grafica tramos consecutivos de 512 muestras (o la ventana que definas) de la señal continua.
    """
    assert is_4d_batches(
        X_test_EOG), "Para continuo, se espera forma [B1,B2,1,512]."
    B1, B2, _, _ = X_test_EOG.shape
    # Construye continuo para clean y denoised:
    noisy_cont = concat_combo(X_test_EOG[combo])            # [B2*512]
    clean_cont = concat_combo(y_test_EOG[combo])            # [B2*512]
    # Para denoised, inferimos por segmentos:
    _, denoised_cont = infer_combo_continuo(X_test_EOG, combo)

    for s in range(desde_seg, min(hasta_seg, B2)):
        i0 = s * ventana
        i1 = i0 + ventana
        plt.figure()
        plt.plot(noisy_cont[i0:i1],  label="Noisy (X)")
        plt.plot(clean_cont[i0:i1],  label="Clean (y)")
        plt.plot(denoised_cont[i0:i1], label="Denoised (ŷ)")
        mse = float(np.mean((denoised_cont[i0:i1] - clean_cont[i0:i1])**2))
        mae = float(np.mean(np.abs(denoised_cont[i0:i1] - clean_cont[i0:i1])))
        plt.title(f"Combo {combo} | Ventana {s}  MSE={mse:.6f} MAE={mae:.6f}")
        plt.xlabel("Muestras (512)")
        plt.ylabel("Amplitud")
        plt.grid(True)
        plt.legend()
        plt.show()


# ===================== EJEMPLOS DE USO =====================
# 1) Graficar los primeros 5 segmentos (512 muestras cada uno) de la combinación 0
# plot_range(X_test_EOG, y_test_EOG, combo=0, seg_ini=0,
#            seg_fin=5, pause=0.0, guardar_png=False)

# 2) Guardar PNG de los segmentos 100..110 de la combinación 3
# plot_range(X_test_EOG, y_test_EOG, combo=3, seg_ini=100, seg_fin=111, pause=0.0, guardar_png=True, carpeta_out="plots_c3")

# 3) Graficar por tramos consecutivos en la señal continua reconstruida (20 ventanas de 512)
plot_continuo_ventanas(X_test_EOG, y_test_EOG, combo=0,
                       ventana=512, desde_seg=0, hasta_seg=20)
