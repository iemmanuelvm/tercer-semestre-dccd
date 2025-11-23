"""
Transformer Denoiser para señales senoidales 1-D
Uso:
  pip install torch matplotlib numpy pandas
  python train_transformer_denoiser.py
"""

import math
import time
import pathlib
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# ----------------- Configuración -----------------
SR = 512                 # Hz
DUR = 2.0                # segundos
SEQ_LEN = int(SR*DUR)    # 1024 muestras
FREQ_RANGE = (2.0, 50.0)  # Hz (frecuencia senoidal aleatoria)
AMP_RANGE = (0.8, 1.2)   # amplitud aleatoria
SNR_DB_RANGE = (0.0, 10.0)  # SNR de entrada en dB

N_TRAIN = 2000
N_VAL = 300
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transformer (pequeño para entrenar rápido en CPU también)
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 128
LR = 2e-4

out_dir = pathlib.Path("./outputs")
out_dir.mkdir(parents=True, exist_ok=True)
model_path = out_dir / "denoiser_transformer.pt"
metrics_csv = out_dir / "training_metrics.csv"
example_npz = out_dir / "example_val_signal.npz"

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ----------------- Generación de datos -----------------


def gen_sinusoid_pair(sr=SR, n=SEQ_LEN, f_range=FREQ_RANGE, amp_range=AMP_RANGE, snr_db_range=SNR_DB_RANGE):
    t = np.arange(n)/sr
    f = np.random.uniform(*f_range)
    amp = np.random.uniform(*amp_range)
    phi = np.random.uniform(0, 2*np.pi)
    clean = amp*np.sin(2*np.pi*f*t + phi).astype(np.float32)

    snr_db = np.random.uniform(*snr_db_range)
    sig_pow = np.mean(clean**2)
    noise_pow = sig_pow / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_pow),
                             size=clean.shape).astype(np.float32)
    noisy = clean + noise
    return clean, noisy


class PairDataset(Dataset):
    def __init__(self, n_items):
        self.clean = []
        self.noisy = []
        for _ in range(n_items):
            c, x = gen_sinusoid_pair()
            self.clean.append(torch.from_numpy(c))
            self.noisy.append(torch.from_numpy(x))

    def __len__(self): return len(self.clean)

    def __getitem__(self, i):
        x = self.noisy[i]  # [T]
        y = self.clean[i]  # [T]
        # Estandarización por muestra (ayuda a entrenar)
        xm = x.mean()
        xs = x.std().clamp_min(1e-6)
        ym = y.mean()
        ys = y.std().clamp_min(1e-6)
        x = (x - xm)/xs
        y = (y - ym)/ys
        return x.unsqueeze(0), y.unsqueeze(0)  # [1, T]

# ----------------- Modelo Transformer 1-D -----------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) *
                        (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B, L, D]
        return x + self.pe[:x.size(1)]


class TransformerDenoiser(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF):
        super().__init__()
        self.inp = nn.Conv1d(1, d_model, kernel_size=7, padding=3)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Conv1d(d_model, 1, kernel_size=7, padding=3)

    def forward(self, x):            # x: [B, 1, T]
        h = self.inp(x).transpose(1, 2)   # [B, L, D]
        h = self.pos(h)
        h = self.enc(h)
        y = self.out(h.transpose(1, 2))   # [B, 1, T]
        return y

# ----------------- Métrica -----------------


def snr_db(clean, est):
    # clean, est: [B, 1, T] o [T]
    if clean.ndim == 1:
        clean = clean[None, None, :]
        est = est[None, None, :]
    noise = clean - est
    sig_pow = (clean**2).mean(dim=-1)
    noise_pow = (noise**2).mean(dim=-1).clamp_min(1e-12)
    snr = 10*torch.log10(sig_pow / noise_pow)
    return snr.mean().item()


def evaluate(model, loader, device):
    model.eval()
    with torch.no_grad():
        snr_in_vals, snr_out_vals = [], []
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = model(noisy)
            snr_in_vals.append(snr_db(clean, noisy))
            snr_out_vals.append(snr_db(clean, pred))
        snr_in = float(np.mean(snr_in_vals))
        snr_out = float(np.mean(snr_out_vals))
        return snr_in, snr_out, (snr_out - snr_in)

# ----------------- Entrenamiento -----------------


def main():
    train_ds = PairDataset(N_TRAIN)
    val_ds = PairDataset(N_VAL)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerDenoiser().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    history = []

    # Línea base antes de entrenar
    snr_in0, snr_out0, snr_i0 = evaluate(model, val_loader, DEVICE)
    print(
        f"Before training | Val SNR_in={snr_in0:.2f} dB, SNR_out={snr_out0:.2f} dB, Improvement={snr_i0:.2f} dB")

    t0 = time.time()
    for ep in range(1, EPOCHS+1):
        model.train()
        total = 0.0
        for noisy, clean in train_loader:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)
            pred = model(noisy)
            loss = 0.7*l1(pred, clean) + 0.3*l2(pred, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()*noisy.size(0)

        snr_in, snr_out, snr_i = evaluate(model, val_loader, DEVICE)
        avg_loss = total / len(train_loader.dataset)
        history.append({"epoch": ep, "train_loss": avg_loss,
                        "val_snr_in": snr_in, "val_snr_out": snr_out, "val_snr_impr": snr_i})
        print(f"Epoch {ep} | train_loss={avg_loss:.5f} | Val SNR_in={snr_in:.2f} dB, "
              f"SNR_out={snr_out:.2f} dB, Improvement={snr_i:.2f} dB")

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.1f} s for {EPOCHS} epochs.")

    # Guardar
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(metrics_csv, index=False)

    # Visualizar un ejemplo de validación
    model.eval()
    with torch.no_grad():
        noisy_ex, clean_ex = val_ds[0]
        den = model(noisy_ex.unsqueeze(0).to(DEVICE)).cpu().squeeze().numpy()

    # Graficar (una sola figura, sin colores específicos)
    noisy_np = noisy_ex.numpy()
    clean_np = clean_ex.numpy()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(clean_np, label="clean")
    plt.plot(noisy_np, label="noisy")
    plt.plot(den, label="denoised")
    plt.legend()
    plt.title("Señal senoidal: clean vs noisy vs denoised")
    plt.xlabel("Muestra")
    plt.ylabel("Amplitud")
    plt.tight_layout()
    plt.show()

    # Guardar ejemplo
    np.savez(example_npz, noisy=noisy_np, clean=clean_np, denoised=den, sr=SR)
    print("Saved model:", str(model_path))
    print("Saved example arrays (.npz):", str(example_npz))
    print("Saved metrics CSV:", str(metrics_csv))


if __name__ == "__main__":
    main()
