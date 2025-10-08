#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd

from utils.model import ResUNetTCN

DEFAULT_CKPT = "./best_joint_denoiser.pt"
DEFAULT_DATA_DIR = "./data/data_for_test"
DEFAULT_OUT_DIR = "./inferences"
DEFAULT_NOISES = ["EMG", "EOG", "CHEW", "SHIV", "ELPP"]
DEFAULT_BS = 256
DEFAULT_SR = 512

torch.manual_seed(42)
np.random.seed(42)

sns.set_style("darkgrid")
plt.rc("font", family="Times New Roman")


def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", None)
    if cfg is None:
        model = ResUNetTCN(in_ch=1, base=64, depth=3, k=7,
                           dropout=0.05, heads=4).to(device)
    else:
        model = ResUNetTCN(**cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def load_test_tensors(data_dir: str, noise: str) -> Tuple[torch.Tensor, torch.Tensor]:
    X = np.load(os.path.join(data_dir, f"X_test_{noise}.npy"))
    y = np.load(os.path.join(data_dir, f"y_test_{noise}.npy"))
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    if X.ndim != 4:
        raise ValueError(
            f"Invalid X_test shape: {X.shape} (expected 4D S,M,C,L)")
    if y.ndim != 4:
        raise ValueError(
            f"Invalid y_test shape: {y.shape} (expected 4D S,M,C,L)")
    return X, y


@torch.inference_mode()
def infer_batched(model: nn.Module, X_test: torch.Tensor, device: torch.device, start_bs: int = DEFAULT_BS, use_autocast: bool = True) -> torch.Tensor:
    S, M, C, L = X_test.shape
    N = S * M
    flat = X_test.reshape(N, C, L)
    bsz_candidates = [start_bs, 128, 64, 32, 16, 8, 4, 2, 1]
    out_cpu = []

    def _clear():
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    for bsz in bsz_candidates:
        try:
            _clear()
            out_cpu = []
            j = 0
            autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16) if (
                use_autocast and device.type == "cuda") else torch.autocast(device_type="cpu", enabled=False))
            with autocast_ctx:
                while j < N:
                    xb = flat[j:j+bsz].to(device, non_blocking=True)
                    yb = model(xb)
                    out_cpu.append(yb.detach().cpu())
                    del xb, yb
                    j += bsz
                    _clear()
            print(f"[INFO] Inference OK with batch_size={bsz}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARN] OOM with batch_size={bsz}, trying smaller…")
                _clear()
                continue
            else:
                raise
    yhat = torch.cat(out_cpu, dim=0).reshape(S, M, C, L)
    return yhat


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    assert y_true.shape == y_pred.shape, f"shape mismatch {y_true.shape} vs {y_pred.shape}"
    y_mean = y_true.mean(axis=-1, keepdims=True)
    p_mean = y_pred.mean(axis=-1, keepdims=True)
    num = np.sum((y_true - y_mean) * (y_pred - p_mean), axis=-1)
    den = np.sqrt(np.sum((y_true - y_mean) ** 2, axis=-1) *
                  np.sum((y_pred - p_mean) ** 2, axis=-1)) + eps
    cc = num / den
    err = y_pred - y_true
    mse = np.mean(err ** 2, axis=-1)
    rmse = np.sqrt(mse)
    rrmse = np.sqrt(np.sum(err ** 2, axis=-1) /
                    (np.sum(y_true ** 2, axis=-1) + eps))
    return {"CC": cc, "MSE": mse, "RMSE": rmse, "RRMSE": rrmse}


def summarize_metrics_per_snr(metrics: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for k, v in metrics.items():
        mean_s = v.mean(axis=(1, 2))
        std_s = v.std(axis=(1, 2), ddof=0)
        out[k] = (mean_s, std_s)
    return out


def overall_means(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {k: float(v.mean()) for k, v in metrics.items()}


def print_metric_summary(name: str, per_snr: Dict[str, Tuple[np.ndarray, np.ndarray]], overall: Dict[str, float]):
    print(f"\n===== {name} =====")
    example_metric = next(iter(per_snr.keys()))
    S = per_snr[example_metric][0].shape[0]
    print("By SNR index:")
    for s in range(S):
        row = []
        for k, (m, sd) in per_snr.items():
            row.append(f"{k}={m[s]:.4f}±{sd[s]:.4f}")
        print(f"  SNR[{s}]: " + " | ".join(row))
    print("Global (mean over S, M, C):")
    print(" | ".join([f"{k}={v:.4f}" for k, v in overall.items()]))


def maybe_save_metrics_csv(noise_name: str, per_snr: Dict[str, Tuple[np.ndarray, np.ndarray]], overall: Dict[str, float], out_dir: str):
    if pd is None:
        print("[INFO] pandas not found; skipping CSV.")
        return
    rows = []
    example_metric = next(iter(per_snr.keys()))
    S = per_snr[example_metric][0].shape[0]
    for s in range(S):
        row = {"noise": noise_name, "snr_idx": s}
        for k, (m, sd) in per_snr.items():
            row[f"{k}_mean"] = float(m[s])
            row[f"{k}_std"] = float(sd[s])
        rows.append(row)
    rows.append({"noise": noise_name, "snr_idx": "ALL", **
                {f"{k}_overall": v for k, v in overall.items()}})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"metrics_{noise_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Metrics CSV saved -> {csv_path}")


def _labels(lang: str) -> Dict[str, str]:
    if lang.lower().startswith("es"):
        return {
            "clean": "EEG limpio",
            "noisy": "EEG con artefactos",
            "denoised": "EEG sin artefactos",
            "time_s": "Tiempo (s)",
            "time_samples": "Tiempo (muestras)",
            "amplitude": "Amplitud",
            "snr_idx": "índice SNR",
            "sample": "muestra",
            "channel": "canal",
            "across_snr": "a través de SNR (horizontal)",
            "train": "Exactitud de Entrenamiento (Fuente)",
            "train_t": "Exactitud de Entrenamiento (Objetivo)",
            "val": "Exactitud de Validación (Fuente)",
            "val_t": "Exactitud de Validación (Objetivo)",
            "title_acc": "Exactitud de Entrenamiento y Validación",
            "epochs": "Épocas",
            "accuracy": "Exactitud",
        }
    else:
        return {
            "clean": "Clean EEG",
            "noisy": "Noisy EEG",
            "denoised": "Denoised EEG",
            "time_s": "Time (s)",
            "time_samples": "Time (samples)",
            "amplitude": "Amplitude",
            "snr_idx": "SNR idx",
            "sample": "sample",
            "channel": "ch",
            "across_snr": "across SNR (horizontal)",
            "train": "Training Accuracy (Source)",
            "train_t": "Training Accuracy (Target)",
            "val": "Validation Accuracy (Source)",
            "val_t": "Validation Accuracy (Target)",
            "title_acc": "Training and Validation Accuracy",
            "epochs": "Epochs",
            "accuracy": "Accuracy",
        }


def plot_eeg_separate(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    yhat: torch.Tensor,
    snr_idx: int = 0,
    sample_idx: int = 0,
    ch: int = 0,
    start: int = 0,
    end: Optional[int] = None,
    title_prefix: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
    sr: int = DEFAULT_SR,
    snr_db: Optional[float] = None,
    language: str = "en",
):
    labels = _labels(language)
    x_np = X_test[snr_idx, sample_idx, ch].detach().cpu().numpy()
    y_np = y_test[snr_idx, sample_idx, ch].detach().cpu().numpy()
    p_np = yhat[snr_idx, sample_idx, ch].detach().cpu().numpy()
    L = x_np.shape[0]
    if end is None or end > L or end < 0:
        end = L
    seg = slice(max(0, start), max(0, end))
    t = np.arange(seg.start, seg.stop) / float(sr)
    m = compute_all_metrics(
        y_np[None, None, None, seg], p_np[None, None, None, seg])
    cc = float(m["CC"][0, 0, 0])
    rmse = float(m["RMSE"][0, 0, 0])
    rrmse = float(m["RRMSE"][0, 0, 0])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, y_np[seg], color="#1f77b4", linewidth=1.2)
    axes[0].set_title(
        f"{title_prefix} {labels['clean']} | SNR[{snr_idx}] ({snr_db:+.0f} dB) {labels['sample']}[{sample_idx}] {labels['channel']}[{ch}]", fontsize=24, fontweight="bold")
    axes[0].set_ylabel(labels["amplitude"], fontsize=24)
    axes[1].plot(t, x_np[seg], color="#d62728", linewidth=1.2)
    axes[1].set_title(
        f"{title_prefix} {labels['noisy']} ({labels['snr_idx']}={snr_idx}, {snr_db:+.0f} dB)", fontsize=24, fontweight="bold")
    axes[1].set_ylabel(labels["amplitude"], fontsize=24)
    axes[2].plot(t, p_np[seg], color="#2ca02c", linewidth=1.2)
    axes[2].set_title(
        f"{title_prefix} {labels['denoised']} | CC={cc:.3f}, RMSE={rmse:.3f}, RRMSE={rrmse:.3f}", fontsize=24, fontweight="bold")
    axes[2].set_xlabel(labels["time_s"], fontsize=24)
    axes[2].set_ylabel(labels["amplitude"], fontsize=24)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figure saved -> {save_path}")
        plt.close()
    elif show:
        plt.show()


def animate_across_snr_horizontal(
    noise_name: str,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    yhat_pred: Optional[torch.Tensor] = None,
    sample_idx: int = 0,
    fps: int = 2,
    show: bool = True,
    language: str = "en",
) -> FuncAnimation:
    labels = _labels(language)
    S, M, C, L = X_test.shape
    assert C == 1, "Expected 1 channel"
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    t = np.arange(L)
    gap = max(1, L // 20)
    chunk = L + gap
    noisy_np = X_test[:, sample_idx, 0, :].detach().cpu().numpy()
    clean_np = y_test[:, sample_idx, 0, :].detach().cpu().numpy()
    pred_np = None
    if yhat_pred is not None:
        _y = yhat_pred
        if torch.is_tensor(_y):
            _y = _y.detach().cpu().numpy()
        pred_np = _y[:, sample_idx, 0, :]
    ymin = float(min(noisy_np.min(), clean_np.min()))
    ymax = float(max(noisy_np.max(), clean_np.max()))
    pad = 0.05 * (ymax - ymin + 1e-6)
    ymin -= pad
    ymax += pad
    fig, ax = plt.subplots()
    (ln_noisy,) = ax.plot([], [], label=labels["noisy"], color="#d62728", linewidth=1.1)
    (ln_pred,) = ax.plot([], [], label=labels["denoised"],
                         color="#2ca02c", linewidth=1.1)
    (ln_clean,) = ax.plot([], [], label=labels["clean"], color="#1f77b4", linewidth=1.1)
    ax.legend(loc="upper right")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, L)
    ax.set_title(
        f"{noise_name} | {labels['sample']} {sample_idx} | {labels['across_snr']}")
    ax.set_xlabel(labels["time_samples"])
    ax.set_ylabel(labels["amplitude"])

    def init():
        ln_noisy.set_data([], [])
        ln_pred.set_data([], [])
        ln_clean.set_data([], [])
        return ln_noisy, ln_pred, ln_clean

    @torch.inference_mode()
    def update(frame):
        offset = frame * chunk
        xb_cpu = X_test[frame, sample_idx].unsqueeze(0)
        yb_cpu = y_test[frame, sample_idx].unsqueeze(0)
        if pred_np is None:
            assert model is not None, "Model required when yhat_pred is None"
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    yhat = model(xb_cpu.to(device)).cpu().squeeze(
                        0).squeeze(0).numpy()
            else:
                yhat = model(xb_cpu).squeeze(0).squeeze(0).numpy()
        else:
            yhat = pred_np[frame]
        x = t + offset
        ln_noisy.set_data(x, xb_cpu.squeeze(
            0).squeeze(0).detach().cpu().numpy())
        ln_pred.set_data(x, yhat)
        ln_clean.set_data(x, yb_cpu.squeeze(
            0).squeeze(0).detach().cpu().numpy())
        ax.set_xlim(offset, offset + L - 1)
        ax.set_xlabel(
            f"{labels['time_samples']} | {labels['snr_idx']} = {frame}/{S-1}")
        return ln_noisy, ln_pred, ln_clean

    anim = FuncAnimation(fig, update, frames=range(
        S), init_func=init, interval=int(1000 / fps), blit=True)
    if show:
        plt.show()
    return anim


def plot_training_metrics(csv_path, language="en", save_path=None, y_min=0.0, y_max=1.0):
    labels = _labels(language)
    plt.rc("font", family="Times New Roman")
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Train Source Acc"], label=labels["train"],
             linewidth=1, linestyle="-", marker="x", markersize=5)
    plt.plot(df.index, df["Train Target Acc"], label=labels["train_t"],
             linewidth=1, linestyle="--", marker="s", markersize=5)
    plt.plot(df.index, df["Val Source Acc"], label=labels["val"],
             linewidth=1, linestyle="-.", marker="^", markersize=5)
    plt.plot(df.index, df["Val Target Acc"], label=labels["val_t"],
             linewidth=1, linestyle=":", marker="d", markersize=5)
    plt.title(labels["title_acc"] + "\n", fontsize=24, fontweight="bold")
    plt.xlabel(labels["epochs"], fontsize=24, fontweight="bold")
    plt.ylabel(labels["accuracy"], fontsize=24, fontweight="bold")
    plt.ylim(y_min, y_max)
    plt.xticks(ticks=range(0, len(df.index) + 5, 5), fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc="lower right", fontsize=24, labelspacing=1.5,
               borderpad=1.5, frameon=True, fancybox=True)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def parse_args():
    ap = argparse.ArgumentParser(
        description="EEG inference, plots, and metrics (CC, MSE, RMSE, RRMSE).")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="Path to model checkpoint (.pt)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="Directory with X_test_*.npy and y_test_*.npy")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="Output directory for predictions/figures/CSV")
    ap.add_argument("--noises", default=",".join(DEFAULT_NOISES),
                    help="Comma-separated list: EMG,EOG,CHEW,SHIV...")
    ap.add_argument("--batch-size", type=int,
                    default=DEFAULT_BS, help="Initial batch size")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--no-autocast", action="store_true",
                    help="Disable autocast (fp16 on CUDA)")
    ap.add_argument("--sample-idx", type=int, default=0,
                    help="Sample index for static plot and animation")
    ap.add_argument("--snr-idx", type=int, default=0,
                    help="SNR index for static plot")
    ap.add_argument("--channel", type=int, default=0,
                    help="Channel for static plot")
    ap.add_argument("--start", type=int, default=0,
                    help="Start sample for static plot")
    ap.add_argument("--end", type=int, default=-1,
                    help="End sample for static plot (-1 = full)")
    ap.add_argument("--show-plot", action="store_true",
                    help="Show the static plot instead of saving")
    ap.add_argument("--no-save-figure", action="store_true",
                    help="Do not save the static figure")
    ap.add_argument("--animate", action="store_true",
                    help="Show horizontal across-SNR animation")
    ap.add_argument("--no-csv", action="store_true",
                    help="Do not save metrics CSV")
    ap.add_argument("--sr", type=int, default=DEFAULT_SR,
                    help="Sampling rate (Hz) for time axis")
    ap.add_argument("--lang", choices=["es", "en"],
                    default="es", help="Language for plot labels")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu else (
        "cuda" if torch.cuda.is_available() else "cpu"))
    use_autocast = (device.type == "cuda") and (not args.no_autocast)
    print(
        f"[INFO] Device: {device} | autocast(fp16)={'ON' if use_autocast else 'OFF'}")
    print(f"[INFO] Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    noises = [n.strip() for n in args.noises.split(",") if n.strip()]
    for noise in noises:
        print(f"\n[INFO] Test inference — {noise}")
        X_test, y_test = load_test_tensors(args.data_dir, noise)
        S, M, C, L = X_test.shape
        print(
            f"  X_test: {tuple(X_test.shape)} | y_test: {tuple(y_test.shape)}")
        yhat_test = infer_batched(
            model, X_test, device, start_bs=args.batch_size, use_autocast=use_autocast)
        pred_out = os.path.join(args.out_dir, f"pred_{noise}.pt")
        torch.save({"yhat": yhat_test, "X_test": X_test,
                   "y_test": y_test}, pred_out)
        print(f"  Predictions saved -> {pred_out}")
        snr_vals = np.linspace(-5, 5, S)
        for s_idx in range(S):
            db = float(snr_vals[s_idx])
            save_path = None if args.no_save_figure else os.path.join(
                args.out_dir, f"separate_{noise}_snr{s_idx}_{db:+.0f}dB.png")
            plot_eeg_separate(
                X_test=X_test,
                y_test=y_test,
                yhat=yhat_test,
                snr_idx=s_idx,
                sample_idx=max(0, min(args.sample_idx, M-1)),
                ch=max(0, min(args.channel, C-1)),
                start=max(0, args.start),
                end=args.end,
                title_prefix=f"{noise} |",
                save_path=save_path,
                show=args.show_plot if save_path is None else False,
                sr=args.sr,
                snr_db=db,
                language=args.lang,
            )
        X_np = X_test.detach().cpu().numpy()
        y_np = y_test.detach().cpu().numpy()
        p_np = yhat_test.detach().cpu().numpy()
        denoised_metrics = compute_all_metrics(y_np, p_np)
        noisy_metrics = compute_all_metrics(y_np, X_np)
        denoised_snr = summarize_metrics_per_snr(denoised_metrics)
        noisy_snr = summarize_metrics_per_snr(noisy_metrics)
        denoised_overall = overall_means(denoised_metrics)
        noisy_overall = overall_means(noisy_metrics)
        print_metric_summary(f"{noise} (DENOISED vs CLEAN)",
                             denoised_snr, denoised_overall)
        print_metric_summary(f"{noise} (NOISY vs CLEAN)",
                             noisy_snr, noisy_overall)
        if not args.no_csv:
            maybe_save_metrics_csv(noise, denoised_snr,
                                   denoised_overall, args.out_dir)
            maybe_save_metrics_csv(noise + "_BASELINE",
                                   noisy_snr, noisy_overall, args.out_dir)
        if args.animate:
            _ = animate_across_snr_horizontal(
                noise_name=noise,
                X_test=X_test,
                y_test=y_test,
                model=model if yhat_test is None else None,
                device=device,
                yhat_pred=yhat_test,
                sample_idx=max(0, min(args.sample_idx, M-1)),
                fps=2,
                show=True,
                language=args.lang,
            )
    print("\n[OK] Inference and metrics complete.")


if __name__ == "__main__":
    main()


# python infer_only_5.py --checkpoint ./best_joint_denoiser.pt --data-dir ./data/data_for_test --out-dir ./inferences --noises ELPP --snr-idx 0 --sample-idx 0 --show-plot
# python infer_only_5.py --checkpoint ./best_joint_denoiser.pt --data-dir ./data/data_for_test --out-dir ./inferences --noises EMG,EOG,SHIV,CHEW --sample-idx 0 --show-plot
# python infer_only_5.py --checkpoint ./best_joint_denoiser.pt --data-dir ./data/data_for_test --out-dir ./inferences --noises EMG,EOG,SHIV,CHEW,ELPP --sample-idx 0 --show-plot --lang en
