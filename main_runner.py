import math
import csv
import random
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data_preparation_runner import prepare_data
from neural_network_runner import Transformer1D

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

X_train = torch.as_tensor(X_train, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.float32)
X_test = torch.as_tensor(X_test,  dtype=torch.float32)
y_test = torch.as_tensor(y_test,  dtype=torch.float32)

# >>> IMPORTANTE: dar forma [B, 1, 512] tanto a train como a test <<<
X_train = X_train.reshape(-1, 1, 512)
y_train = y_train.reshape(-1, 1, 512)
X_test = X_test.reshape(-1, 1, 512)
y_test = y_test.reshape(-1, 1, 512)

print(f"X_train {X_train.shape}")
print(f"y_train {y_train.shape}")
print(f"X_test  {X_test.shape}")
print(f"y_test  {y_test.shape}")

# ---------------------------
# Dataloaders
use_cuda = torch.cuda.is_available()
common_loader_kwargs = dict(
    pin_memory=use_cuda, num_workers=0 if use_cuda else 0)

# Split de validación desde el train (p.ej., 10%)
full_train = TensorDataset(X_train, y_train)
val_frac = 0.10
val_len = int(len(full_train) * val_frac)
train_len = len(full_train) - val_len
train_ds, val_ds = random_split(full_train, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=64,
                          shuffle=True,  drop_last=True,  **common_loader_kwargs)
valid_loader = DataLoader(val_ds,   batch_size=64,
                          shuffle=False, drop_last=False, **common_loader_kwargs)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64,
                         shuffle=False, drop_last=False, **common_loader_kwargs)

# ---------------------------
# Modelo
device = torch.device("cuda" if use_cuda else "cpu")
model = Transformer1D(
    in_ch=1, out_ch=1,
    d_model=192, depth=4, n_heads=6,
    patch_size=4, ff_mult=3, dropout=0.2,
    causal=False, use_residual_in_out=True
).to(device)
print(
    f"Parameters (M): {sum(p.numel() for p in model.parameters()) / 1e6:.6f}")

# ---------------------------
# Optimizador + LR schedule (CAWR) + AMP + clipping
criterion = nn.SmoothL1Loss(beta=0.5)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, weight_decay=1e-3, betas=(0.9, 0.95))

# CosineAnnealingWarmRestarts: evita LR=0 con eta_min
scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=3e-6)

use_amp = use_cuda  # AMP solo si hay CUDA
scaler = GradScaler(enabled=use_amp)

# ---------------------------
# Métricas
EPS = 1e-12


def update_running_stats(batch_pred: torch.Tensor, batch_true: torch.Tensor, acc: Dict[str, float]):
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


def finalize_metrics(acc: Dict[str, float]):
    n = max(1, acc["n"])
    mse = acc["mse_sum"] / n
    rmse = math.sqrt(max(mse, 0.0))
    mae = acc["mae_sum"] / n
    rrmse = rmse / (acc["sum_abs_t"] / n + EPS)
    num = n * acc["sum_pt"] - acc["sum_p"] * acc["sum_t"]
    den = math.sqrt(max((n * acc["sum_p2"] - acc["sum_p"] ** 2)
                    * (n * acc["sum_t2"] - acc["sum_t"] ** 2), 0.0)) + EPS
    cc = num / den
    return mse, rmse, rrmse, mae, cc


def fresh_acc():
    return {k: 0.0 for k in ["mse_sum", "mae_sum", "sum_p", "sum_t", "sum_p2", "sum_t2", "sum_pt", "sum_abs_t", "n"]}


def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    acc = fresh_acc()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                pred = model(xb)
            update_running_stats(pred, yb, acc)
    return finalize_metrics(acc)


# ---------------------------
# Entrenamiento
EPOCHS = 300
best_val = float("inf")
best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

metrics_log = []

patience = 30
bad_epochs = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    acc_tr = fresh_acc()

    for step, (xb, yb) in enumerate(train_loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # Sanitizar batch
        if not (torch.isfinite(xb).all() and torch.isfinite(yb).all()):
            continue

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            pred = model(xb)
            loss = criterion(pred, yb)

        # Backward con AMP + clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Scheduler por batch con epoch fraccional
        scheduler.step(epoch + step / max(1, len(train_loader)))

        update_running_stats(pred, yb, acc_tr)

    train_mse, train_rmse, train_rrmse, train_mae, train_cc = finalize_metrics(
        acc_tr)

    # --------- Validación --------- #
    val_mse, val_rmse, val_rrmse, val_mae, val_cc = evaluate(
        model, valid_loader)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch:03d} | "
        f"train MSE {train_mse:.6f} RMSE {train_rmse:.6f} RRMSE {train_rrmse:.6f} CC {train_cc:.4f} | "
        f"val MSE {val_mse:.6f} RMSE {val_rmse:.6f} RRMSE {val_rrmse:.6f} CC {val_cc:.4f} | "
        f"lr {current_lr:.2e} | grad_norm {float(grad_norm):.2f}"
    )

    # Log por época
    metrics_log.append({
        "epoch": epoch,
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "train_rrmse": train_rrmse,
        "train_mae": train_mae,
        "train_cc": train_cc,
        "val_mse": val_mse,
        "val_rmse": val_rmse,
        "val_rrmse": val_rrmse,
        "val_mae": val_mae,
        "val_cc": val_cc,
        "lr": float(current_lr),
        "grad_norm": float(grad_norm),
    })

    # Early stopping + checkpoint best
    if val_mse < best_val - 1e-5:
        best_val = val_mse
        best_state = {k: v.detach().cpu()
                      for k, v in model.state_dict().items()}
        bad_epochs = 0
    else:
        bad_epochs += 1

    if bad_epochs >= patience:
        print(
            f"Early stopping en epoch {epoch} (sin mejora {patience} épocas).")
        break

# Guardar mejor modelo (según val)
torch.save(best_state, "best_Transformer1D_cc_rrmse_emg.pth")

# --------- Evaluación en TEST con el mejor estado --------- #
model.load_state_dict(best_state, strict=True)
test_mse, test_rmse, test_rrmse, test_mae, test_cc = evaluate(
    model, test_loader)
print(
    f"[TEST] MSE {test_mse:.6f} RMSE {test_rmse:.6f} RRMSE {test_rrmse:.6f} "
    f"MAE {test_mae:.6f} CC {test_cc:.4f}"
)

# ---------------------------
# Guardar métricas a CSV (incluye fila final de test)
csv_path = "metrics_log.csv"
if len(metrics_log) > 0:
    fieldnames = list(metrics_log[0].keys())
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_log:
            writer.writerow(row)
    print(f"Métricas de entrenamiento/validación guardadas en: {csv_path}")

    # Añadir fila de test al final (sin epoch numérico)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "epoch": "TEST(best)",
            "train_mse": "",
            "train_rmse": "",
            "train_rrmse": "",
            "train_mae": "",
            "train_cc": "",
            "val_mse": "",
            "val_rmse": "",
            "val_rrmse": "",
            "val_mae": "",
            "val_cc": "",
            "lr": "",
            "grad_norm": "",
        })
    print("Fila de TEST agregada al CSV (como separador).")

    # Guardar métricas de test en un CSV aparte (opcional)
    with open("metrics_test_best.csv", mode="w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["mse", "rmse", "rrmse", "mae", "cc"])
        writer.writeheader()
        writer.writerow({
            "mse":   test_mse,
            "rmse":  test_rmse,
            "rrmse": test_rrmse,
            "mae":   test_mae,
            "cc":    test_cc
        })
    print("Métricas de TEST(best) guardadas en: metrics_test_best.csv")
else:
    print("No hay métricas para guardar.")
