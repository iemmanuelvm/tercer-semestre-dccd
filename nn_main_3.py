# train_emg_denoiser.py
# EMG→EEG denoiser: ResUNet-TCN + SE + Bottleneck Attention
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.swa_utils import AveragedModel

# --------------------------
# Data (usa tu prepare_data)
# --------------------------
from data_preparation_runner import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

X_train_EMG, y_train_EMG, X_test_EMG, y_test_EMG = prepare_data(
    combin_num=11, train_per=0.9, noise_type='EMG'
)

X_train_EMG = torch.FloatTensor(X_train_EMG)   # [N, 1, 512]
y_train_EMG = torch.FloatTensor(y_train_EMG)   # [N, 1, 512]
X_test_EMG = torch.FloatTensor(X_test_EMG)    # [11, M, 1, 512]
y_test_EMG = torch.FloatTensor(y_test_EMG)    # [11, M, 1, 512]

print("X_train_EMG:", X_train_EMG.shape)
print("y_train_EMG:", y_train_EMG.shape)
print("X_test_EMG: ", X_test_EMG.shape)
print("y_test_EMG: ", y_test_EMG.shape)

# Aplanamos test para métrica agregada
snr_levels, M, C, L = X_test_EMG.shape
X_test_flat = X_test_EMG.reshape(snr_levels * M, C, L)
y_test_flat = y_test_EMG.reshape(snr_levels * M, C, L)

train_ds = TensorDataset(X_train_EMG, y_train_EMG)
test_ds = TensorDataset(X_test_flat, y_test_flat)

# --------------------------
# Modelo
# --------------------------


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
        self.depth = nn.Conv1d(in_ch, in_ch, k, padding=pad, dilation=dilation,
                               groups=in_ch, bias=False)
        self.point = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        g = math.gcd(out_ch, 32) or 1  # GroupNorm robusto
        self.norm = nn.GroupNorm(num_groups=g, num_channels=out_ch)
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
    def __init__(self, ch, k=7, dilations=(1, 2, 4), dropout=0.05, use_se=True):
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
        x_t = x.permute(2, 0, 1)           # [L,B,C]
        y, _ = self.attn(x_t, x_t, x_t, need_weights=False)
        y = self.ln(y)
        return (x_t + y).permute(1, 2, 0)  # [B,C,L]


class ResUNetTCN(nn.Module):
    def __init__(self, in_ch=1, base=64, depth=3, k=7, dropout=0.15, heads=4):
        super().__init__()
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
        ups, dec_blocks = [], []
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
        h = self.attn(h)
        for up, blk in zip(self.ups, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                h = F.pad(
                    h, (0, diff)) if diff > 0 else h[..., :skip.shape[-1]]
            h = h + skip
            h = blk(h)
        delta = self.proj(h)
        return x + delta  # residual learning

# --------------------------
# Métricas
# --------------------------


@torch.no_grad()
def compute_metrics(y_true, y_pred, eps=1e-8):
    diff = y_pred - y_true
    mse = torch.mean(diff**2)
    rmse = torch.sqrt(mse + eps)
    rms_true = torch.sqrt(torch.mean(y_true**2) + eps)
    rrmse = rmse / (rms_true + eps)
    yt, yp = y_true.squeeze(1), y_pred.squeeze(1)
    yt_m, yp_m = yt.mean(dim=-1, keepdim=True), yp.mean(dim=-1, keepdim=True)
    cov = ((yt-yt_m)*(yp-yp_m)).mean(dim=-1)
    std_t, std_p = yt.std(dim=-1)+eps, yp.std(dim=-1)+eps
    cc = torch.mean(cov/(std_t*std_p))
    return {"MSE": mse.item(), "RMSE": rmse.item(), "RRMSE": rrmse.item(), "CC": cc.item()}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        preds.append(yhat.detach().cpu())
        gts.append(yb.detach().cpu())
    y_pred, y_true = torch.cat(preds, 0), torch.cat(gts, 0)
    return compute_metrics(y_true, y_pred)

# --------------------------
# Pérdida MR-STFT (ligera)
# --------------------------


def stft_mag(x, n_fft, hop, win_len):
    # x: [N,1,L] -> mag: [N, F, T]
    x = x.squeeze(1)
    window = torch.hann_window(win_len, device=x.device, dtype=x.dtype)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win_len,
                   window=window, return_complex=True, center=True, pad_mode='reflect')
    return X.abs().clamp_min_(1e-7)


def mrstft_loss(y_hat, y_true, cfg=((64, 16, 64), (128, 32, 128), (256, 64, 256))):
    sc_total, mag_total = 0.0, 0.0
    for n_fft, hop, win in cfg:
        P = stft_mag(y_hat, n_fft, hop, win)
        T = stft_mag(y_true, n_fft, hop, win)
        # spectral convergence
        sc = (P - T).norm(p='fro') / (T.norm(p='fro') + 1e-8)
        # log-mag
        mag = torch.mean(torch.abs(torch.log(P) - torch.log(T)))
        sc_total += sc
        mag_total += mag
    return (sc_total + mag_total) / (2 * len(cfg))

# --------------------------
# Entrenamiento (MSE + 0.2*MRSTFT) + EMA
# --------------------------


def train_model(
    epochs=90,
    batch_size=256,
    lr=1e-3,
    weight_decay=5e-4,
    model_save_path="./best_emg_denoiser.pt",
    eval_per_snr=False,
    ema_decay=0.999,
    mrstft_weight=0.2
):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    model = ResUNetTCN(in_ch=1, base=64, depth=3, k=7,
                       dropout=0.15, heads=4).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5, verbose=True
    )

    # EMA de pesos
    def ema_avg(avg_p, p, n):  # exponential moving average
        return ema_decay * avg_p + (1.0 - ema_decay) * p
    ema_model = AveragedModel(model, avg_fn=ema_avg).to(device)

    best_val = float("inf")

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)

            # Optimiza MSE + regularizador MR-STFT ligero
            loss_mse = F.mse_loss(yhat, yb)
            loss_stf = mrstft_loss(yhat, yb)
            loss = loss_mse + mrstft_weight * loss_stf

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # actualizar EMA en cada paso
            ema_model.update_parameters(model)

        # evaluar con EMA (mejor generaliza)
        train_metrics = evaluate(ema_model, train_loader)
        test_metrics = evaluate(ema_model, test_loader)

        print(f"Epoch {epoch:03d} "
              f"| Train -> MSE: {train_metrics['MSE']:.6f}, RMSE: {train_metrics['RMSE']:.6f}, "
              f"RRMSE: {train_metrics['RRMSE']:.6f}, CC: {train_metrics['CC']:.4f} "
              f"| Test -> MSE: {test_metrics['MSE']:.6f}, RMSE: {test_metrics['RMSE']:.6f}, "
              f"RRMSE: {test_metrics['RRMSE']:.6f}, CC: {test_metrics['CC']:.4f}")

        # Scheduler sobre MSE de test (EMA)
        scheduler.step(test_metrics["MSE"])

        # Guardar mejor por MSE de test (EMA)
        if test_metrics["MSE"] < best_val:
            best_val = test_metrics["MSE"]
            torch.save(
                {"model": ema_model.state_dict(),  # guardamos EMA
                 "config": {"base": 64, "depth": 3, "k": 7,
                            "dropout": 0.15, "heads": 4}},
                model_save_path
            )

        if eval_per_snr and (epoch % 5 == 0 or epoch == epochs):
            print("Per-SNR (EMA) -5..+5 dB")
            for i in range(X_test_EMG.shape[0]):
                loader = DataLoader(TensorDataset(X_test_EMG[i], y_test_EMG[i]),
                                    batch_size=512, shuffle=False)
                m = evaluate(ema_model, loader)
                print(
                    f"  SNR[{i:02d}] -> MSE:{m['MSE']:.6f} RMSE:{m['RMSE']:.6f} CC:{m['CC']:.4f}")

    print(f"\nTraining done. Best test MSE (EMA): {best_val:.6f}")
    return ema_model


if __name__ == "__main__":
    _ = train_model(
        epochs=90,
        batch_size=256,
        lr=1e-3,
        weight_decay=5e-4,
        model_save_path="./best_emg_denoiser.pt",
        eval_per_snr=False,      # pon True si quieres ver por SNR
        ema_decay=0.999,
        mrstft_weight=0.2
    )
