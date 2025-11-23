import os
import gc
import math
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils.model import ResUNetTCN

DEFAULT_CKPT = "./best_joint_denoiser.pt"
DEFAULT_DATA_DIR = "./data/data_for_test"
DEFAULT_OUT_DIR = "./inferences"
DEFAULT_NOISES = ["EMG", "EOG", "CHEW", "SHIV"]
DEFAULT_BS = 256

torch.manual_seed(42)
np.random.seed(42)


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
            f"X_test shape inválido: {X.shape} (esperado 4D S,M,C,L)")
    if y.ndim != 4:
        raise ValueError(
            f"y_test shape inválido: {y.shape} (esperado 4D S,M,C,L)")
    return X, y


@torch.inference_mode()
def infer_batched(
    model: nn.Module,
    X_test: torch.Tensor,
    device: torch.device,
    start_bs: int = DEFAULT_BS,
    use_autocast: bool = True
) -> torch.Tensor:
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
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if (use_autocast and device.type == "cuda")
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                while j < N:
                    xb = flat[j:j+bsz].to(device, non_blocking=True)
                    yb = model(xb)
                    out_cpu.append(yb.detach().cpu())
                    del xb, yb
                    j += bsz
                    _clear()
            print(f"[INFO] Inference OK con batch_size={bsz}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARN] OOM con batch_size={bsz}, probando menor…")
                _clear()
                continue
            else:
                raise

    yhat = torch.cat(out_cpu, dim=0).reshape(S, M, C, L)
    return yhat


try:
    import pandas as pd
except Exception:
    pd = None


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12
) -> Dict[str, np.ndarray]:
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
    for k, v in metrics.items():  # (S, M, C)
        mean_s = v.mean(axis=(1, 2))
        std_s = v.std(axis=(1, 2), ddof=0)
        out[k] = (mean_s, std_s)
    return out


def overall_means(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {k: float(v.mean()) for k, v in metrics.items()}


def print_metric_summary(name: str, per_snr: Dict[str, Tuple[np.ndarray, np.ndarray]], overall: Dict[str, float]):
    print(f"\n===== {name} =====")
    # por SNR
    example_metric = next(iter(per_snr.keys()))
    S = per_snr[example_metric][0].shape[0]
    print("Por índice de SNR:")
    for s in range(S):
        row = []
        for k, (m, sd) in per_snr.items():
            row.append(f"{k}={m[s]:.4f}±{sd[s]:.4f}")
        print(f"  SNR[{s}]: " + " | ".join(row))
    print("Global (promedio sobre S, M, C):")
    print(" | ".join([f"{k}={v:.4f}" for k, v in overall.items()]))


def maybe_save_metrics_csv(
    noise_name: str,
    per_snr: Dict[str, Tuple[np.ndarray, np.ndarray]],
    overall: Dict[str, float],
    out_dir: str,
):
    if pd is None:
        print("[INFO] pandas no encontrado; omitiendo CSV.")
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
    print(f"[INFO] CSV de métricas guardado -> {csv_path}")


def plot_eeg_triplet(
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
):
    x_np = X_test[snr_idx, sample_idx, ch].detach().cpu().numpy()
    y_np = y_test[snr_idx, sample_idx, ch].detach().cpu().numpy()
    p_np = yhat[snr_idx, sample_idx, ch].detach().cpu().numpy()
    L = x_np.shape[0]
    if end is None or end > L or end < 0:
        end = L
    seg = slice(max(0, start), max(0, end))
    t = np.arange(seg.start, seg.stop)

    m = compute_all_metrics(
        y_np[None, None, None, seg], p_np[None, None, None, seg])
    cc = float(m["CC"][0, 0, 0])
    rmse = float(m["RMSE"][0, 0, 0])
    rrmse = float(m["RRMSE"][0, 0, 0])

    plt.figure(figsize=(12, 4))
    plt.plot(t, x_np[seg], label="contaminated", alpha=0.8)
    plt.plot(t, p_np[seg], label="denoised (inference)", alpha=0.9)
    plt.plot(t, y_np[seg], label="clean (target)", alpha=0.9)
    ttl = f"{title_prefix} SNR[{snr_idx}] sample[{sample_idx}] ch[{ch}] | CC={cc:.3f}, RMSE={rmse:.3f}, RRMSE={rrmse:.3f}"
    plt.title(ttl)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figura guardada -> {save_path}")
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
) -> FuncAnimation:
    S, M, C, L = X_test.shape
    assert C == 1, "Se esperaba 1 canal"
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
    (ln_noisy,) = ax.plot([], [], label="noisy")
    (ln_pred,) = ax.plot([], [], label="denoised")
    (ln_clean,) = ax.plot([], [], label="clean")
    ax.legend(loc="upper right")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, L)
    ax.set_title(
        f"{noise_name} | sample {sample_idx} | across SNR (horizontal)")
    ax.set_xlabel("Tiempo (samples)")
    ax.set_ylabel("Amplitud")

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
            assert model is not None, "Se requiere modelo si no hay yhat_pred"
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
        ax.set_xlabel(f"Tiempo (samples) | SNR idx = {frame}/{S-1}")
        return ln_noisy, ln_pred, ln_clean

    anim = FuncAnimation(fig, update, frames=range(
        S), init_func=init, interval=int(1000 / fps), blit=True)
    if show:
        plt.show()
    return anim


def parse_args():
    ap = argparse.ArgumentParser(
        description="Inferencia EEG + gráficos + métricas (CC, MSE, RMSE, RRMSE).")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="Ruta al checkpoint del modelo (.pt)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="Directorio con X_test_*.npy y y_test_*.npy")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="Directorio de salida para predicciones/figuras/CSV")
    ap.add_argument("--noises", default=",".join(DEFAULT_NOISES),
                    help="Lista separada por comas: EMG,EOG,CHEW,SHIV...")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BS,
                    help="Tamaño de lote inicial")
    ap.add_argument("--cpu", action="store_true", help="Forzar CPU")
    ap.add_argument("--no-autocast", action="store_true",
                    help="Desactivar autocast (fp16 en CUDA)")
    ap.add_argument("--sample-idx", type=int, default=0,
                    help="Índice de muestra para el gráfico estático y animación")
    ap.add_argument("--snr-idx", type=int, default=0,
                    help="Índice de SNR para el gráfico estático")
    ap.add_argument("--channel", type=int, default=0,
                    help="Canal para el gráfico estático")
    ap.add_argument("--start", type=int, default=0,
                    help="Sample inicial para el gráfico estático")
    ap.add_argument("--end", type=int, default=-1,
                    help="Sample final para el gráfico estático (-1 = todo)")
    ap.add_argument("--show-plot", action="store_true",
                    help="Mostrar el gráfico estático (en lugar de guardarlo)")
    ap.add_argument("--no-save-triplet", action="store_true",
                    help="No guardar la figura estática")
    ap.add_argument("--animate", action="store_true",
                    help="Mostrar animación horizontal across SNR")
    ap.add_argument("--no-csv", action="store_true",
                    help="No guardar CSV de métricas")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu" if args.cpu else (
        "cuda" if torch.cuda.is_available() else "cpu"))
    use_autocast = (device.type == "cuda") and (not args.no_autocast)

    print(
        f"[INFO] Dispositivo: {device} | autocast(fp16)={'ON' if use_autocast else 'OFF'}")
    print(f"[INFO] Cargando modelo desde: {args.ckpt}")
    model = load_model(args.ckpt, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    noises = [n.strip() for n in args.noises.split(",") if n.strip()]

    for noise in noises:
        print(f"\n[INFO] Inferencia en TEST — {noise}")
        X_test, y_test = load_test_tensors(args.data_dir, noise)
        S, M, C, L = X_test.shape
        print(
            f"  X_test: {tuple(X_test.shape)} | y_test: {tuple(y_test.shape)}")

        yhat_test = infer_batched(
            model, X_test, device, start_bs=args.batch_size, use_autocast=use_autocast)

        pred_out = os.path.join(args.out_dir, f"pred_{noise}.pt")
        torch.save({"yhat": yhat_test, "X_test": X_test,
                   "y_test": y_test}, pred_out)
        print(f"  Predicciones guardadas -> {pred_out}")

        save_path = None if args.no_save_triplet else os.path.join(
            args.out_dir, f"triplet_{noise}.png")
        plot_eeg_triplet(
            X_test=X_test, y_test=y_test, yhat=yhat_test,
            snr_idx=max(0, min(args.snr_idx, S-1)),
            sample_idx=max(0, min(args.sample_idx, M-1)),
            ch=max(0, min(args.channel, C-1)),
            start=max(0, args.start),
            end=args.end,
            title_prefix=f"{noise} |",
            save_path=save_path,
            show=args.show_plot if save_path is None else False
        )

        X_np = X_test.detach().cpu().numpy()
        y_np = y_test.detach().cpu().numpy()
        p_np = yhat_test.detach().cpu().numpy()

        denoised_metrics = compute_all_metrics(y_np, p_np)
        noisy_metrics = compute_all_metrics(
            y_np, X_np)

        denoised_snr = summarize_metrics_per_snr(denoised_metrics)
        noisy_snr = summarize_metrics_per_snr(noisy_metrics)

        denoised_overall = overall_means(denoised_metrics)
        noisy_overall = overall_means(noisy_metrics)

        print_metric_summary(f"{noise} (DENOISED vs CLEAN)",
                             denoised_snr, denoised_overall)
        print_metric_summary(
            f"{noise} (NOISY vs CLEAN)   ", noisy_snr,    noisy_overall)

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
            )

    print("\n[OK] Inferencia + métricas completadas.")


if __name__ == "__main__":
    main()
