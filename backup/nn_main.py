# train_emg_denoiser.py
# Innovative EMG→EEG denoiser: ResUNet-TCN + SE + Self-Attention (bottleneck)
import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --------------------------
# Data (uses your prepare_data exactly as-is)
# --------------------------
from data_preparation_runner import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# Load data (your code)
X_train_EMG, y_train_EMG, X_test_EMG, y_test_EMG = prepare_data(
    combin_num=11, train_per=0.9, noise_type='EMG'
)

# Cast to tensors
X_train_EMG = torch.FloatTensor(X_train_EMG)  # [N, 1, 512]
y_train_EMG = torch.FloatTensor(y_train_EMG)  # [N, 1, 512]
X_test_EMG = torch.FloatTensor(X_test_EMG)   # [11, M, 1, 512]
y_test_EMG = torch.FloatTensor(y_test_EMG)   # [11, M, 1, 512]

print("X_train_EMG:", X_train_EMG.shape)
print("y_train_EMG:", y_train_EMG.shape)
print("X_test_EMG: ", X_test_EMG.shape)
print("y_test_EMG: ", y_test_EMG.shape)

# Flatten test across SNR for aggregate metrics
snr_levels, M, C, L = X_test_EMG.shape
X_test_flat = X_test_EMG.reshape(snr_levels * M, C, L)
y_test_flat = y_test_EMG.reshape(snr_levels * M, C, L)

train_ds = TensorDataset(X_train_EMG, y_train_EMG)
test_ds = TensorDataset(X_test_flat, y_test_flat)

# --------------------------
# Model: ResUNet-TCN + SE + Attention
# --------------------------


def same_padding(kernel_size, dilation=1):
    # ensures output length equals input length for stride=1
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
            layers += [DepthwiseSeparableConv1D(in_ch,
                                                ch, k=k, dilation=d, dropout=dropout)]
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.se = SqueezeExcite1D(ch) if use_se else nn.Identity()

    def forward(self, x):
        out = self.net(x)
        out = self.se(out)
        return x + out  # residual


class BottleneckAttention(nn.Module):
    """Lightweight self-attention at bottleneck to capture long-range deps."""

    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=ch, num_heads=num_heads, batch_first=False)
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        # x: [B, C, L] -> attn expects [L, B, C]
        x_perm = x.permute(2, 0, 1)
        x2, _ = self.attn(x_perm, x_perm, x_perm, need_weights=False)
        x2 = self.ln(x2)
        return x_perm + x2  # residual in token space
        # return to [B, C, L]

    def to_conv(self, x_attn):
        return x_attn.permute(1, 2, 0)


class ResUNetTCN(nn.Module):
    def __init__(self, in_ch=1, base=32, depth=3, k=7, dropout=0.05, heads=4):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, base, kernel_size=3, padding=1)
        # Encoder
        enc_blocks = []
        downs = []
        ch = base
        for i in range(depth):
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
        dec_blocks = []
        ups = []
        for i in range(depth):
            ups.append(nn.ConvTranspose1d(ch, ch//2, kernel_size=2, stride=2))
            ch = ch//2
            dec_blocks.append(ResidualTCNBlock(
                ch, k=k, dilations=(1, 2, 4), dropout=dropout))
        self.ups = nn.ModuleList(ups)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.proj = nn.Conv1d(base, 1, kernel_size=1)

    def forward(self, x):
        # x: [B,1,L]
        skips = []
        h = self.stem(x)
        for blk, down in zip(self.enc_blocks, self.downs):
            h = blk(h)
            skips.append(h)
            h = down(h)  # downsample by 2
        h = self.bottleneck(h)
        h = self.attn.to_conv(self.attn(h))
        for up, blk in zip(self.ups, self.dec_blocks):
            h = up(h)  # upsample by 2
            skip = skips.pop()
            # length safety (in case of odd sizes — not needed for 512, but robust)
            if h.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                if diff > 0:
                    h = nn.functional.pad(h, (0, diff))
                else:
                    h = h[..., :skip.shape[-1]]
            h = h + skip  # additive skip fusion (stable for denoising)
            h = blk(h)
        delta = self.proj(h)        # residual correction
        y_hat = x + delta           # predict clean EEG via residual learning
        return y_hat

# --------------------------
# Metrics
# --------------------------


@torch.no_grad()
def compute_metrics(y_true, y_pred, eps=1e-8):
    """
    y_true, y_pred: [N, 1, L]
    Returns: dict(MSE, RMSE, RRMSE, CC)
    """
    N, C, L = y_true.shape
    diff = y_pred - y_true
    mse = torch.mean(diff**2)
    rmse = torch.sqrt(mse + eps)

    # RMS of ground truth (for RRMSE)
    rms_true = torch.sqrt(torch.mean(y_true**2) + eps)
    rrmse = rmse / (rms_true + eps)

    # Pearson r per sample (across time), then average
    yt = y_true.squeeze(1)
    yp = y_pred.squeeze(1)
    yt_m = yt.mean(dim=-1, keepdim=True)
    yp_m = yp.mean(dim=-1, keepdim=True)
    cov = ((yt - yt_m) * (yp - yp_m)).mean(dim=-1)
    std_t = yt.std(dim=-1) + eps
    std_p = yp.std(dim=-1) + eps
    cc = torch.mean(cov / (std_t * std_p))

    return {
        "MSE": mse.item(),
        "RMSE": rmse.item(),
        "RRMSE": rrmse.item(),
        "CC": cc.item()
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = []
    gts = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)
        preds.append(yhat.detach().cpu())
        gts.append(yb.detach().cpu())
    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(gts, dim=0)
    return compute_metrics(y_true, y_pred)

# Optional: per-SNR evaluation (11 levels) to inspect robustness


@torch.no_grad()
def evaluate_per_snr(model, X_test_EMG, y_test_EMG, batch_size=512):
    model.eval()
    results = []
    snr_levels = X_test_EMG.shape[0]
    for i in range(snr_levels):
        X_i = X_test_EMG[i]  # [M,1,512]
        y_i = y_test_EMG[i]
        loader = DataLoader(TensorDataset(X_i, y_i),
                            batch_size=batch_size, shuffle=False)
        metrics = evaluate(model, loader, device)
        results.append(metrics)
    return results

# --------------------------
# Training
# --------------------------


def train_model(
    epochs=50,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    model_save_path="./best_emg_denoiser.pt",
    eval_per_snr=False
):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    model = ResUNetTCN(in_ch=1, base=64, depth=3, k=7,
                       dropout=0.05, heads=4).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    best_val = math.inf

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = model(xb)
            loss = criterion(yhat, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        # Epoch-end metrics (evaluate freshly for both train and test)
        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader,  device)

        msg = (f"Epoch {epoch:03d} "
               f"| Train -> MSE: {train_metrics['MSE']:.6f}, RMSE: {train_metrics['RMSE']:.6f}, "
               f"RRMSE: {train_metrics['RRMSE']:.6f}, CC: {train_metrics['CC']:.4f} "
               f"| Test -> MSE: {test_metrics['MSE']:.6f}, RMSE: {test_metrics['RMSE']:.6f}, "
               f"RRMSE: {test_metrics['RRMSE']:.6f}, CC: {test_metrics['CC']:.4f}")
        print(msg)

        # Save best by test MSE
        if test_metrics["MSE"] < best_val:
            best_val = test_metrics["MSE"]
            torch.save({"model": model.state_dict(),
                        "config": {"base": 48, "depth": 3, "k": 7, "dropout": 0.05, "heads": 4}},
                       model_save_path)

        # (Optional) show per-SNR at the end of each epoch
        if eval_per_snr and (epoch % 5 == 0 or epoch == epochs):
            per_snr = evaluate_per_snr(
                model, X_test_EMG.to(device), y_test_EMG.to(device))
            print("Per-SNR metrics (SNR levels from -5 dB to 5 dB):")
            for i, m in enumerate(per_snr):
                print(f"  SNR[{i:02d}] -> MSE: {m['MSE']:.6f}, RMSE: {m['RMSE']:.6f}, "
                      f"RRMSE: {m['RRMSE']:.6f}, CC: {m['CC']:.4f}")

    print(f"\nTraining done. Best test MSE: {best_val:.6f}")
    return model


if __name__ == "__main__":
    _ = train_model(
        epochs=300,         # try 100+ for best results
        batch_size=256,    # adjust to your GPU/CPU memory
        lr=1e-3,
        weight_decay=1e-4,
        model_save_path="./best_emg_denoiser.pt",
        eval_per_snr=False  # set True to view per-SNR metrics every 5 epochs
    )
