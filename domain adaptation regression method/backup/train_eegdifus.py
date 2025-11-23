# train_eegdifus.py
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
from data_preparation_runner import prepare_data
from eeg_difus import (
    EEGDfusDenoiser,
    make_beta_schedule,
    diffusion_training_step,
    ddpm_sample,
)

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
# Dataloaders
use_cuda = torch.cuda.is_available()
common_loader_kwargs = dict(pin_memory=use_cuda, num_workers=0)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True,
    drop_last=True,
    **common_loader_kwargs
)

valid_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=64,
    shuffle=False,
    drop_last=False,
    **common_loader_kwargs
)

# ---------------------------
# Modelo
device = torch.device("cuda" if use_cuda else "cpu")
model = EEGDfusDenoiser(
    in_ch=1, hidden_dim=64, heads=1, qkv_dim=32, depth=3, dropout=0.1
).to(device)

print(
    f"Parameters (M): {sum(p.numel() for p in model.parameters()) / 1e6:.6f}")

# ---------------------------
# Optimizador + LR schedule (warmup + cosine) + AMP + clipping
base_lr = 3e-4
optimizer = torch.optim.AdamW(
    model.parameters(), lr=base_lr, weight_decay=1e-2, betas=(0.9, 0.95)
)

EPOCHS = 500
steps_per_epoch = len(train_loader)
total_steps = max(1, EPOCHS * steps_per_epoch)
warmup_steps = max(1, int(0.1 * total_steps))  # 10% warmup


def lr_lambda(step_idx: int):
    if step_idx < warmup_steps:
        return (step_idx + 1) / warmup_steps
    progress = (step_idx - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# Difusión: schedule
T = 500
betas, alphas, alpha_bars = make_beta_schedule(
    T=T, beta_start=1e-4, beta_end=2e-2)
betas, alphas, alpha_bars = betas.to(
    device), alphas.to(device), alpha_bars.to(device)

use_amp = use_cuda
scaler = GradScaler(enabled=use_amp)

# ---------------------------
# Métricas
EPS = 1e-12


def update_running_stats(batch_pred, batch_true, acc):
    p = batch_pred.view(-1).detach().cpu()
    t = batch_true.view(-1).detach().cpu()
    diff = p - t
    acc["mse_sum"] += (diff.pow(2).sum()).item()
    acc["mae_sum"] += (diff.abs().sum()).item()
    acc["sum_p"] += p.sum().item()
    acc["sum_t"] += t.sum().item()
    acc["sum_p2"] += (p.pow(2).sum()).item()
    acc["sum_t2"] += (t.pow(2).sum()).item()
    acc["sum_pt"] += (p * t).sum().item()
    acc["sum_abs_t"] += t.abs().sum().item()
    acc["n"] += p.numel()


def finalize_metrics(acc):
    n = max(1, acc["n"])
    mse = acc["mse_sum"] / n
    rmse = math.sqrt(max(mse, 0.0))
    mae = acc["mae_sum"] / n
    rrmse = rmse / (acc["sum_abs_t"] / n + EPS)
    num = n * acc["sum_pt"] - acc["sum_p"] * acc["sum_t"]
    den = math.sqrt(max((n * acc["sum_p2"] - acc["sum_p"] ** 2) *
                        (n * acc["sum_t2"] - acc["sum_t"] ** 2), 0.0)) + EPS
    cc = num / den
    return mse, rmse, rrmse, mae, cc


def fresh_acc():
    return {k: 0.0 for k in ["mse_sum", "mae_sum", "sum_p", "sum_t", "sum_p2", "sum_t2", "sum_pt", "sum_abs_t", "n"]}


# ---------------------------
# Entrenamiento
best_val = float("inf")
best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
global_step = 0

for epoch in range(1, EPOCHS + 1):
    # --------- TRAIN (difusión) --------- #
    model.train()
    run_tr_loss = 0.0
    seen = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)  # x̃ (ruidosa)
        yb = yb.to(device, non_blocking=True)  # x0 (limpia)

        if not (torch.isfinite(xb).all() and torch.isfinite(yb).all()):
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            loss = diffusion_training_step(
                model, x0=yb, x_tilde=xb, alpha_bars=alpha_bars
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        global_step += 1
        bs = xb.size(0)
        run_tr_loss += float(loss) * bs
        seen += bs

    train_loss = run_tr_loss / max(1, seen)

    # --------- VALIDACIÓN (sampling DDPM) --------- #
    model.eval()
    acc_val = fresh_acc()
    with torch.no_grad():
        # Para acelerar validación, submuestrea pasos
        T_SAMPLE = 100
        if T_SAMPLE < len(betas):
            idx = torch.linspace(0, len(betas) - 1, T_SAMPLE).long().to(device)
            betas_v = betas[idx]
            alphas_v = alphas[idx]
            alpha_bars_v = alpha_bars[idx]
        else:
            betas_v, alphas_v, alpha_bars_v = betas, alphas, alpha_bars

        for xb, yb in valid_loader:
            xb = xb.to(device, non_blocking=True)  # x̃
            yb = yb.to(device, non_blocking=True)  # x0

            y_hat = ddpm_sample(
                model, x_tilde=xb,
                betas=betas_v, alphas=alphas_v, alpha_bars=alpha_bars_v,
                device=device
            )
            update_running_stats(y_hat, yb, acc_val)

    val_mse, val_rmse, val_rrmse, val_mae, val_cc = finalize_metrics(acc_val)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:03d} | "
        f"train loss {train_loss:.6f} | "
        f"val MSE {val_mse:.6f} RMSE {val_rmse:.6f} RRMSE {val_rrmse:.6f} CC {val_cc:.4f} | "
        f"lr {current_lr:.2e} | grad_norm {float(grad_norm):.2f}"
    )

    if val_mse < best_val:
        best_val = val_mse
        best_state = {k: v.detach().cpu()
                      for k, v in model.state_dict().items()}

# Guardado
torch.save(best_state, "best_EEGDfus_DDPM.pth")
print("Guardado: best_EEGDfus_DDPM.pth")
