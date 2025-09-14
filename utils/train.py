import os
import csv
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.model import ResUNetTCN
from utils.metrics import evaluate, evaluate_per_snr
from utils.safe_io import load_target_pt_folders

try:
    from geomloss import SamplesLoss
except ImportError as e:
    raise ImportError(
        "GeomLoss is not installed. Install with: pip install geomloss") from e


def kl_divergence(p, q, eps: float = 1e-6):
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return torch.sum(p * torch.log(p / q), dim=-1)


def jensen_shannon_divergence(p, q, eps: float = 1e-6):
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)
    return jsd


def jensen_shannon_distance(p, q, eps: float = 1e-6):
    jsd = jensen_shannon_divergence(p, q, eps)
    jsd = torch.clamp(jsd, min=0.0)
    return torch.sqrt(jsd)


def _maybe_init_csv(path: str, fieldnames: list[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def _append_row(path: str, row: dict, fieldnames: list[str]):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def _to_probs(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    min_vals = torch.amax(-x, dim=dim, keepdim=True)
    min_vals = torch.clamp(min_vals, min=0.0)
    x_shift = x + min_vals + eps
    denom = torch.sum(x_shift, dim=dim, keepdim=True) + eps
    return x_shift / denom


@torch.no_grad()
def _compute_js_kl_over_loader(model, loader, device, use_softmax: bool = False):
    model.eval()
    kl_vals, jsd_vals, jsdist_vals = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)

        if use_softmax:
            p = torch.softmax(yhat, dim=-1)
            q = torch.softmax(yb, dim=-1)
        else:
            p = _to_probs(yhat, dim=-1)
            q = _to_probs(yb,   dim=-1)

        kl_batch = kl_divergence(p, q).mean().item()
        jsd_batch = jensen_shannon_divergence(p, q).mean().item()
        jsdist_batch = jensen_shannon_distance(p, q).mean().item()

        kl_vals.append(kl_batch)
        jsd_vals.append(jsd_batch)
        jsdist_vals.append(jsdist_batch)

    return {
        "KL": float(sum(kl_vals) / max(1, len(kl_vals))),
        "JSD": float(sum(jsd_vals) / max(1, len(jsd_vals))),
        "JSDist": float(sum(jsdist_vals) / max(1, len(jsdist_vals))),
    }


def train_model(
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    L: int,
    device: torch.device,
    X_test_EMG=None,
    y_test_EMG=None,
    X_test_EOG=None,
    y_test_EOG=None,
    epochs=120,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    model_save_path="./best_joint_denoiser.pt",
    eval_per_snr=False,
    domain_adapt=True,
    target_dirs=("./eyem", "./musc"),
    ot_kind="sinkhorn",
    ot_weight=0.1,
    ramp_epochs=20,
    ot_blur=0.05,
    feature_l2norm=True,
    metrics_csv_path="trainig_metrics.csv",
    js_use_softmax=False
):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, drop_last=False,
        pin_memory=torch.cuda.is_available()
    )

    X_target = load_target_pt_folders(target_dirs, L)
    if X_target.numel() == 0:
        domain_adapt = False
        print("[WARN] No valid target data found. Training without adaptation.")
        target_loader = None
    else:
        target_ds = TensorDataset(X_target)
        target_loader = DataLoader(
            target_ds, batch_size=batch_size, shuffle=True, drop_last=False,
            pin_memory=torch.cuda.is_available()
        )

    model = ResUNetTCN(in_ch=1, base=64, depth=3, k=7,
                       dropout=0.05, heads=4).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    if ot_kind not in ("sinkhorn", "energy"):
        raise ValueError("ot_kind must be 'sinkhorn' or 'energy'")
    samples_loss = SamplesLoss(
        loss=ot_kind, p=2, blur=ot_blur, debias=True if ot_kind == "sinkhorn" else False
    )

    fieldnames = [
        "epoch", "lr",
        "train_MSE", "train_RMSE", "train_RRMSE", "train_CC",
        "test_MSE",  "test_RMSE",  "test_RRMSE",  "test_CC",
        "train_KL", "train_JSD", "train_JSDist",
        "test_KL",  "test_JSD",  "test_JSDist",
        "ot_lambda_alpha"
    ]
    _maybe_init_csv(metrics_csv_path, fieldnames)

    best_val = math.inf
    print("\n[INFO] Start joint training (EMG+EOG) with OT adaptation =",
          ot_kind if domain_adapt else "disabled")

    if domain_adapt:
        tgt_iter = iter(target_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            yhat, f_src = model(xb, return_features=True)
            sup_loss = criterion(yhat, yb)

            if domain_adapt:
                try:
                    xt = next(tgt_iter)[0]
                except StopIteration:
                    tgt_iter = iter(target_loader)
                    xt = next(tgt_iter)[0]
                xt = xt.to(device)
                with torch.set_grad_enabled(True):
                    f_tgt = model.encode(xt)
                if feature_l2norm:
                    f_src = nn.functional.normalize(f_src, dim=1)
                    f_tgt = nn.functional.normalize(f_tgt, dim=1)
                ot_loss = samples_loss(f_src, f_tgt)
                lam = min(1.0, epoch / max(1, ramp_epochs))
                total_loss = sup_loss + lam * ot_weight * ot_loss
            else:
                ot_loss = torch.tensor(0.0, device=device)
                lam = 0.0
                total_loss = sup_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader,  device)

        train_js = _compute_js_kl_over_loader(
            model, train_loader, device, use_softmax=js_use_softmax)
        test_js = _compute_js_kl_over_loader(
            model, test_loader,  device, use_softmax=js_use_softmax)

        msg = (
            f"Epoch {epoch:03d} | "
            f"Train -> MSE {train_metrics['MSE']:.6f}, RMSE {train_metrics['RMSE']:.6f}, "
            f"RRMSE {train_metrics['RRMSE']:.6f}, CC {train_metrics['CC']:.4f} | "
            f"Test -> MSE {test_metrics['MSE']:.6f}, RMSE {test_metrics['RMSE']:.6f}, "
            f"RRMSE {test_metrics['RRMSE']:.6f}, CC {test_metrics['CC']:.4f} | "
            f"KL/JSD/JSDist (train) {train_js['KL']:.4f}/{train_js['JSD']:.4f}/{train_js['JSDist']:.4f} | "
            f"(test) {test_js['KL']:.4f}/{test_js['JSD']:.4f}/{test_js['JSDist']:.4f}"
        )
        if domain_adapt:
            msg += f" | OT λ*α={lam*ot_weight:.3f}"
        print(msg)

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_MSE": train_metrics["MSE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_RRMSE": train_metrics["RRMSE"],
            "train_CC": train_metrics["CC"],
            "test_MSE": test_metrics["MSE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_RRMSE": test_metrics["RRMSE"],
            "test_CC": test_metrics["CC"],
            "train_KL": train_js["KL"],
            "train_JSD": train_js["JSD"],
            "train_JSDist": train_js["JSDist"],
            "test_KL": test_js["KL"],
            "test_JSD": test_js["JSD"],
            "test_JSDist": test_js["JSDist"],
            "ot_lambda_alpha": (lam * ot_weight) if domain_adapt else 0.0,
        }
        _append_row(metrics_csv_path, row, fieldnames)

        if test_metrics["MSE"] < best_val:
            best_val = test_metrics["MSE"]
            torch.save({"model": model.state_dict(),
                       "config": model.cfg}, model_save_path)

        if eval_per_snr and (epoch % 10 == 0 or epoch == epochs):
            if X_test_EMG is not None and y_test_EMG is not None:
                print("\n[INFO] Per-SNR EMG:")
                per_snr_emg = evaluate_per_snr(
                    model, X_test_EMG, y_test_EMG, device)
                for i, m in enumerate(per_snr_emg):
                    print(
                        f"  EMG SNR[{i:02d}] -> MSE {m['MSE']:.6f}, RMSE {m['RMSE']:.6f}, RRMSE {m['RRMSE']:.6f}, CC {m['CC']:.4f}")
            if X_test_EOG is not None and y_test_EOG is not None:
                print("\n[INFO] Per-SNR EOG:")
                per_snr_eog = evaluate_per_snr(
                    model, X_test_EOG, y_test_EOG, device)
                for i, m in enumerate(per_snr_eog):
                    print(
                        f"  EOG SNR[{i:02d}] -> MSE {m['MSE']:.6f}, RMSE {m['RMSE']:.6f}, RRMSE {m['RRMSE']:.6f}, CC {m['CC']:.4f}")
            print("")

    print(f"\n[OK] Training finished. Best test MSE: {best_val:.6f}")
    return model
