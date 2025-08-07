import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler
from data_preparation_runner import prepare_data
from neural_network_runner import UNet1D

X_train, y_train, X_test, y_test = prepare_data(
    combin_num=11,
    train_per=0.9,
    noise_type="EMG",
)

X_train = torch.as_tensor(X_train, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.float32)
X_test = torch.as_tensor(X_test,  dtype=torch.float32)
y_test = torch.as_tensor(y_test,  dtype=torch.float32)

X_test = X_test.reshape(-1, 1, 512)
y_test = y_test.reshape(-1, 1, 512)

print(f"X_train {X_train.shape}")
print(f"y_train {y_train.shape}")
print(f"X_test  {X_test.shape}")
print(f"y_test  {y_test.shape}")

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True,
    drop_last=True,
)

valid_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=64,
    shuffle=False,
    drop_last=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet1D(base=64, dropout=0.1).to(device)
print(
    f"Parameters (M): {sum(p.numel() for p in model.parameters()) / 1e6:.6f}")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)

EPS = 1e-12


def update_running_stats(batch_pred, batch_true, acc):
    p = batch_pred.view(-1).detach().cpu()
    t = batch_true.view(-1).detach().cpu()
    bs = p.numel()

    diff = p - t

    acc["mse_sum"] += (diff.pow(2).sum()).item()
    acc["mae_sum"] += (diff.abs().sum()).item()

    acc["sum_p"] += p.sum().item()
    acc["sum_t"] += t.sum().item()
    acc["sum_p2"] += (p.pow(2).sum()).item()
    acc["sum_t2"] += (t.pow(2).sum()).item()
    acc["sum_pt"] += (p * t).sum().item()
    acc["sum_abs_t"] += t.abs().sum().item()
    acc["n"] += bs


def finalize_metrics(acc):
    n = acc["n"]
    mse = acc["mse_sum"] / n
    rmse = math.sqrt(mse)
    mae = acc["mae_sum"] / n
    rrmse = rmse / (acc["sum_abs_t"] / n + EPS)

    num = n * acc["sum_pt"] - acc["sum_p"] * acc["sum_t"]
    den = math.sqrt(
        (n * acc["sum_p2"] - acc["sum_p"] ** 2) *
        (n * acc["sum_t2"] - acc["sum_t"] ** 2) + EPS
    )
    cc = num / den if den > 0 else 0.0

    return mse, rmse, rrmse, mae, cc


EPOCHS = 500
use_amp = torch.cuda.is_available()
scaler = GradScaler(enabled=use_amp)

best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    # --------- Train phase --------- #
    model.train()
    acc_tr = {k: 0.0 for k in [
        "mse_sum", "mae_sum", "sum_p", "sum_t", "sum_p2", "sum_t2", "sum_pt", "sum_abs_t", "n"]}

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            pred = model(xb)
            loss = criterion(pred, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        update_running_stats(pred, yb, acc_tr)

    train_mse, train_rmse, train_rrmse, train_mae, train_cc = finalize_metrics(
        acc_tr)

    model.eval()
    acc_val = {k: 0.0 for k in acc_tr}

    with torch.no_grad():
        for xb, yb in valid_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                pred = model(xb)
            update_running_stats(pred, yb, acc_val)

    val_mse, val_rmse, val_rrmse, val_mae, val_cc = finalize_metrics(acc_val)

    scheduler.step(val_mse)

    print(
        f"Epoch {epoch:03d} | "
        f"train MSE {train_mse:.6f} RMSE {train_rmse:.6f} RRMSE {train_rrmse:.6f} CC {train_cc:.4f} | "
        f"val MSE {val_mse:.6f} RMSE {val_rmse:.6f} RRMSE {val_rrmse:.6f} CC {val_cc:.4f}"
    )

    if val_mse < best_val:
        best_val = val_mse
        best_state = {k: v.detach().cpu()
                      for k, v in model.state_dict().items()}

torch.save(best_state, "best_unet1d_cc_rrmse_emg.pth")
