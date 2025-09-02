# final_inference.py
# Real-time (window-by-window) inference animation for EMG/EOG denoising
# Place this file in: C:\my-dccd\eeg_denoised\final_inference.py

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------
# Data loader from your project
# ------------------------------------------------------------
from data_preparation_runner import prepare_data

# ------------------------------------------------------------
# Repro & device
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------
# Helpers used in your training script
# ------------------------------------------------------------


def to_tensor(x):
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 2:
        t = t.unsqueeze(1)
    return t


def same_padding(kernel_size, dilation=1):
    return ((kernel_size - 1) * dilation) // 2

# ------------------------------------------------------------
# Model (exact same architecture used during training)
# ------------------------------------------------------------


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
        # x: [B,C,L] -> [L,B,C]
        x_lbc = x.permute(2, 0, 1)
        y, _ = self.attn(x_lbc, x_lbc, x_lbc, need_weights=False)
        y = self.ln(y)
        y = x_lbc + y
        return y

    def to_conv(self, x_attn):
        return x_attn.permute(1, 2, 0)  # [B,C,L]


class ResUNetTCN(nn.Module):
    def __init__(self, in_ch=1, base=64, depth=3, k=7, dropout=0.05, heads=4):
        super().__init__()
        self.cfg = dict(in_ch=in_ch, base=base, depth=depth,
                        k=k, dropout=dropout, heads=heads)
        self.stem = nn.Conv1d(in_ch, base, kernel_size=3, padding=1)
        # Encoder
        enc_blocks, downs = [], []
        ch = base
        for _ in range(depth):
            enc_blocks.append(ResidualTCNBlock(
                ch, k=k, dilations=(1, 2, 4), dropout=dropout))
            downs.append(nn.Conv1d(ch, ch*2, kernel_size=2, stride=2))
            ch *= 2
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downs = nn.ModuleList(downs)
        # Bottleneck
        self.bottleneck = ResidualTCNBlock(
            ch, k=k, dilations=(1, 2, 4, 8), dropout=dropout)
        self.attn = BottleneckAttention(ch, num_heads=heads)
        # Decoder
        dec_blocks, ups = [], []
        for _ in range(depth):
            ups.append(nn.ConvTranspose1d(ch, ch//2, kernel_size=2, stride=2))
            ch = ch//2
            dec_blocks.append(ResidualTCNBlock(
                ch, k=k, dilations=(1, 2, 4), dropout=dropout))
        self.ups = nn.ModuleList(ups)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.proj = nn.Conv1d(base, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        h = self.stem(x)
        for blk, down in zip(self.enc_blocks, self.downs):
            h = blk(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        h = self.attn.to_conv(self.attn(h))
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
        return x + delta

# ------------------------------------------------------------
# Metrics (optional)
# ------------------------------------------------------------


@torch.no_grad()
def compute_metrics(y_true, y_pred, eps=1e-8):
    diff = y_pred - y_true
    mse = torch.mean(diff**2)
    rmse = torch.sqrt(mse + eps)
    rms_true = torch.sqrt(torch.mean(y_true**2) + eps)
    rrmse = rmse / (rms_true + eps)

    yt = y_true.squeeze(1)
    yp = y_pred.squeeze(1)
    yt_m = yt.mean(dim=-1, keepdim=True)
    yp_m = yp.mean(dim=-1, keepdim=True)
    cov = ((yt - yt_m) * (yp - yp_m)).mean(dim=-1)
    std_t = yt.std(dim=-1) + eps
    std_p = yp.std(dim=-1) + eps
    cc = torch.mean(cov / (std_t * std_p))

    return {"MSE": mse.item(), "RMSE": rmse.item(), "RRMSE": rrmse.item(), "CC": cc.item()}

# ------------------------------------------------------------
# Loading & animation utilities
# ------------------------------------------------------------


def load_model(model_path: str) -> ResUNetTCN:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt.get("config", {"in_ch": 1, "base": 64,
                   "depth": 3, "k": 7, "dropout": 0.05, "heads": 4})
    model = ResUNetTCN(**cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_window(model: nn.Module, x_1c_512: torch.Tensor) -> np.ndarray:
    """
    x_1c_512: [1,1,512] float32 tensor on any device
    returns: numpy [512]
    """
    x_1c_512 = x_1c_512.to(DEVICE)
    yhat = model(x_1c_512)
    return yhat.detach().cpu().numpy().squeeze()


def to_np_1d(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().squeeze()


def estimate_limits(model, X_emg_snr, X_eog_snr, num_frames, n_probe=10):
    n = min(n_probe, num_frames)
    vals = []
    for i in range(n):
        # EMG
        x_emg_t = X_emg_snr[i:i+1]
        x_emg = to_np_1d(x_emg_t[0, 0])
        yhat_emg = predict_window(model, x_emg_t)
        vals.extend([x_emg, yhat_emg, x_emg - yhat_emg])
        # EOG
        x_eog_t = X_eog_snr[i:i+1]
        x_eog = to_np_1d(x_eog_t[0, 0])
        yhat_eog = predict_window(model, x_eog_t)
        vals.extend([x_eog, yhat_eog, x_eog - yhat_eog])
    allv = np.concatenate(vals, axis=0)
    vmax = np.percentile(np.abs(allv), 99)
    return (-1.1 * vmax, 1.1 * vmax)


def cc_metric(a, b, eps=1e-8):
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.std() + eps) * (b.std() + eps)
    return float(((a*b).mean() / denom))

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Real-time denoise animation (EMG + EOG)")
    parser.add_argument("--model-path", type=str, default="./best_joint_denoiser.pt",
                        help="Ruta al checkpoint entrenado (.pt)")
    parser.add_argument("--snr-index", type=int, default=0,
                        help="Índice SNR a visualizar (0..SNR-1)")
    parser.add_argument("--interval-ms", type=int, default=120,
                        help="Intervalo entre frames en milisegundos")
    parser.add_argument("--save-as", type=str, default=None, choices=[None, "gif", "mp4"],
                        help="Guardar la animación (gif/mp4). Por defecto: None (solo muestra)")
    parser.add_argument("--save-name", type=str, default="realtime_denoise_animation",
                        help="Nombre base de archivo al guardar")
    args = parser.parse_args()

    print("[INFO] Cargando modelo:", args.model_path)
    model = load_model(args.model_path)

    # --------------------------
    # Prepare test data (EMG/EOG)
    # --------------------------
    print("[INFO] Preparando datos de prueba (EMG/EOG)...")
    # EMG
    X_train_EMG, y_train_EMG, X_test_EMG, y_test_EMG = prepare_data(
        combin_num=11, train_per=0.9, noise_type="EMG"
    )
    # EOG
    X_train_EOG, y_train_EOG, X_test_EOG, y_test_EOG = prepare_data(
        combin_num=11, train_per=0.9, noise_type="EOG"
    )

    X_test_EMG = to_tensor(X_test_EMG)
    y_test_EMG = to_tensor(y_test_EMG)
    X_test_EOG = to_tensor(X_test_EOG)
    y_test_EOG = to_tensor(y_test_EOG)

    assert X_test_EMG.ndim == 4 and y_test_EMG.ndim == 4, "Test EMG debe ser [SNR,M,1,512]"
    assert X_test_EOG.ndim == 4 and y_test_EOG.ndim == 4, "Test EOG debe ser [SNR,M,1,512]"
    assert X_test_EMG.shape[-1] == 512 and X_test_EOG.shape[-1] == 512, "La ventana debe ser de 512 muestras"

    SNR = X_test_EMG.shape[0]
    if not (0 <= args.snr_index < SNR):
        raise IndexError(
            f"SNR_INDEX fuera de rango. Recibido {args.snr_index}, disponible 0..{SNR-1}")

    X_emg_snr = X_test_EMG[args.snr_index]  # [M,1,512]
    X_eog_snr = X_test_EOG[args.snr_index]  # [M,1,512]

    num_frames = int(min(X_emg_snr.shape[0], X_eog_snr.shape[0]))
    window_len = X_emg_snr.shape[-1]
    x_axis = np.arange(window_len)

    # --------------------------
    # Figure setup
    # --------------------------
    fig, (ax_emg, ax_eog) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.25}
    )

    # EMG lines
    (emg_base_line,) = ax_emg.plot([], [], lw=1, label="Base (X)")
    (emg_clean_line,) = ax_emg.plot([], [], lw=1, label="Noise-removed (ŷ)")
    (emg_noise_line,) = ax_emg.plot([], [], lw=1, label="Noise (X−ŷ)")
    ax_emg.set_title(f"EMG — SNR index {args.snr_index}")
    ax_emg.set_ylabel("Amplitud")
    ax_emg.legend(loc="upper right")

    # EOG lines
    (eog_base_line,) = ax_eog.plot([], [], lw=1, label="Base (X)")
    (eog_clean_line,) = ax_eog.plot([], [], lw=1, label="Noise-removed (ŷ)")
    (eog_noise_line,) = ax_eog.plot([], [], lw=1, label="Noise (X−ŷ)")
    ax_eog.set_title(f"EOG — SNR index {args.snr_index}")
    ax_eog.set_xlabel("Muestras (512)")
    ax_eog.set_ylabel("Amplitud")
    ax_eog.legend(loc="upper right")

    # Robust shared y-limits (from early frames)
    ymin, ymax = estimate_limits(
        model, X_emg_snr, X_eog_snr, num_frames, n_probe=10)
    ax_emg.set_ylim(ymin, ymax)
    ax_eog.set_ylim(ymin, ymax)
    ax_emg.set_xlim(0, window_len - 1)
    ax_eog.set_xlim(0, window_len - 1)

    @torch.no_grad()
    def init():
        for ln in (emg_base_line, emg_clean_line, emg_noise_line,
                   eog_base_line, eog_clean_line, eog_noise_line):
            ln.set_data([], [])
        return (emg_base_line, emg_clean_line, emg_noise_line,
                eog_base_line, eog_clean_line, eog_noise_line)

    @torch.no_grad()
    def update(frame_idx):
        # EMG
        x_emg_t = X_emg_snr[frame_idx:frame_idx+1]  # [1,1,512]
        x_emg = to_np_1d(x_emg_t[0, 0])
        yhat_emg = predict_window(model, x_emg_t)
        noise_emg = x_emg - yhat_emg

        emg_base_line.set_data(x_axis, x_emg)
        emg_clean_line.set_data(x_axis, yhat_emg)
        emg_noise_line.set_data(x_axis, noise_emg)

        try:
            cc_emg = cc_metric(x_emg, yhat_emg)
            ax_emg.set_title(
                f"EMG — SNR index {args.snr_index} | frame {frame_idx+1}/{num_frames} | CC≈{cc_emg:.3f}")
        except Exception:
            pass

        # EOG
        x_eog_t = X_eog_snr[frame_idx:frame_idx+1]
        x_eog = to_np_1d(x_eog_t[0, 0])
        yhat_eog = predict_window(model, x_eog_t)
        noise_eog = x_eog - yhat_eog

        eog_base_line.set_data(x_axis, x_eog)
        eog_clean_line.set_data(x_axis, yhat_eog)
        eog_noise_line.set_data(x_axis, noise_eog)

        try:
            cc_eog = cc_metric(x_eog, yhat_eog)
            ax_eog.set_title(
                f"EOG — SNR index {args.snr_index} | frame {frame_idx+1}/{num_frames} | CC≈{cc_eog:.3f}")
        except Exception:
            pass

        return (emg_base_line, emg_clean_line, emg_noise_line,
                eog_base_line, eog_clean_line, eog_noise_line)

    # NOTE: blit=False tends to be more compatible across Windows backends
    anim = FuncAnimation(
        fig, update, frames=num_frames, init_func=init,
        interval=args.interval_ms, blit=False, repeat=True
    )

    plt.tight_layout()

    if args.save_as is None:
        # Show interactive window
        plt.show()
    else:
        # Save animation
        save_name = args.save_name
        if args.save_as == "mp4":
            # Requires ffmpeg available in PATH
            anim.save(f"{save_name}.mp4", writer="ffmpeg", dpi=150)
            print(f"[OK] Guardado {save_name}.mp4")
        elif args.save_as == "gif":
            anim.save(f"{save_name}.gif", writer="pillow", dpi=150)
            print(f"[OK] Guardado {save_name}.gif")


if __name__ == "__main__":
    main()
