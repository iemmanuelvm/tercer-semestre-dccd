import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_preparation_runner import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# EMG
X_train_EMG, y_train_EMG, X_test_EMG, y_test_EMG = prepare_data(
    combin_num=11, train_per=0.9, noise_type="EMG"
)
# EOG
X_train_EOG, y_train_EOG, X_test_EOG, y_test_EOG = prepare_data(
    combin_num=11, train_per=0.9, noise_type="EOG"
)

# --------------------------
#   X_train_*: [N, 1, 512]
#   y_train_*: [N, 1, 512]
#   X_test_* : [SNR=11, M, 1, 512]
#   y_test_* : [SNR=11, M, 1, 512]
# --------------------------


def to_tensor(x):
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 2:
        t = t.unsqueeze(1)
    return t


X_train_EMG = to_tensor(X_train_EMG)
y_train_EMG = to_tensor(y_train_EMG)
X_test_EMG = to_tensor(X_test_EMG)
y_test_EMG = to_tensor(y_test_EMG)

X_train_EOG = to_tensor(X_train_EOG)
y_train_EOG = to_tensor(y_train_EOG)
X_test_EOG = to_tensor(X_test_EOG)
y_test_EOG = to_tensor(y_test_EOG)

assert X_train_EMG.ndim == 3 and y_train_EMG.ndim == 3, "Train EMG debe ser [N,1,L]"
assert X_train_EOG.ndim == 3 and y_train_EOG.ndim == 3, "Train EOG debe ser [N,1,L]"
assert X_test_EMG.ndim == 4 and y_test_EMG.ndim == 4, "Test EMG debe ser [SNR,M,1,L]"
assert X_test_EOG.ndim == 4 and y_test_EOG.ndim == 4, "Test EOG debe ser [SNR,M,1,L]"
assert X_train_EMG.shape[1] == 1 and X_train_EOG.shape[1] == 1, "Canales debe ser 1"
L = X_train_EMG.shape[-1]
assert L == X_train_EOG.shape[-1] == y_train_EMG.shape[-1] == y_train_EOG.shape[-1], "Longitudes deben coincidir"

print(f"[INFO] Longitud de ventana L={L}")
print("X_train_EMG:", tuple(X_train_EMG.shape))
print("y_train_EMG:", tuple(y_train_EMG.shape))
print("X_test_EMG :", tuple(X_test_EMG.shape))
print("y_test_EMG :", tuple(y_test_EMG.shape))

print("X_train_EOG:", tuple(X_train_EOG.shape))
print("y_train_EOG:", tuple(y_train_EOG.shape))
print("X_test_EOG :", tuple(X_test_EOG.shape))
print("y_test_EOG :", tuple(y_test_EOG.shape))

X_train_joint = torch.cat([X_train_EMG, X_train_EOG], dim=0)
y_train_joint = torch.cat([y_train_EMG, y_train_EOG], dim=0)


def flatten_snr(X4, y4):
    SNR, M, C, L = X4.shape
    return X4.reshape(SNR * M, C, L), y4.reshape(SNR * M, C, L)


X_test_EMG_flat, y_test_EMG_flat = flatten_snr(X_test_EMG, y_test_EMG)
X_test_EOG_flat, y_test_EOG_flat = flatten_snr(X_test_EOG, y_test_EOG)

X_test_joint = torch.cat([X_test_EMG_flat, X_test_EOG_flat], dim=0)
y_test_joint = torch.cat([y_test_EMG_flat, y_test_EOG_flat], dim=0)

train_ds = TensorDataset(X_train_joint, y_train_joint)
test_ds = TensorDataset(X_test_joint,  y_test_joint)

print("[INFO] Joint train:", tuple(X_train_joint.shape),
      " | Joint test:", tuple(X_test_joint.shape))


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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)
        preds.append(yhat.detach().cpu())
        gts.append(yb.detach().cpu())
    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(gts, dim=0)
    return compute_metrics(y_true, y_pred)


@torch.no_grad()
def evaluate_per_snr(model, X_test_4D, y_test_4D, device, batch_size=512):
    model.eval()
    SNR = X_test_4D.shape[0]
    out = []
    for i in range(SNR):
        ds = TensorDataset(X_test_4D[i], y_test_4D[i])
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        out.append(evaluate(model, dl, device))
    return out

# --------------------------
# Train
# --------------------------


def train_model(
    epochs=120,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    model_save_path="./best_joint_denoiser.pt",
    eval_per_snr=False
):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, drop_last=False,
        pin_memory=torch.cuda.is_available()
    )

    model = ResUNetTCN(in_ch=1, base=64, depth=3, k=7,
                       dropout=0.05, heads=4).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    best_val = math.inf
    print("\n[INFO] Inicio de entrenamiento conjunto (EMG+EOG)...")
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

        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader,  device)
        print(
            f"Epoch {epoch:03d} | "
            f"Train -> MSE {train_metrics['MSE']:.6f}, RMSE {train_metrics['RMSE']:.6f}, "
            f"RRMSE {train_metrics['RRMSE']:.6f}, CC {train_metrics['CC']:.4f} | "
            f"Test -> MSE {test_metrics['MSE']:.6f}, RMSE {test_metrics['RMSE']:.6f}, "
            f"RRMSE {test_metrics['RRMSE']:.6f}, CC {test_metrics['CC']:.4f}"
        )

        if test_metrics["MSE"] < best_val:
            best_val = test_metrics["MSE"]
            torch.save(
                {"model": model.state_dict(), "config": model.cfg},
                model_save_path
            )

        if eval_per_snr and (epoch % 10 == 0 or epoch == epochs):
            print("\n[INFO] Per-SNR EMG:")
            per_snr_emg = evaluate_per_snr(
                model, X_test_EMG, y_test_EMG, device)
            for i, m in enumerate(per_snr_emg):
                print(
                    f"  EMG SNR[{i:02d}] -> MSE {m['MSE']:.6f}, RMSE {m['RMSE']:.6f}, RRMSE {m['RRMSE']:.6f}, CC {m['CC']:.4f}")

            print("\n[INFO] Per-SNR EOG:")
            per_snr_eog = evaluate_per_snr(
                model, X_test_EOG, y_test_EOG, device)
            for i, m in enumerate(per_snr_eog):
                print(
                    f"  EOG SNR[{i:02d}] -> MSE {m['MSE']:.6f}, RMSE {m['RMSE']:.6f}, RRMSE {m['RRMSE']:.6f}, CC {m['CC']:.4f}")
            print("")

    print(f"\n[OK] Entrenamiento finalizado. Best test MSE: {best_val:.6f}")
    return model


if __name__ == "__main__":
    _ = train_model(
        epochs=100,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        model_save_path="./best_joint_denoiser.pt",
        eval_per_snr=False
    )
