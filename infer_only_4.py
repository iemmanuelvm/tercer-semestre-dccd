import os
import gc
import csv
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import torch
import torch.nn as nn

from utils.model import ResUNetTCN


def load_mat_as_dict(path):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def unwrap_mat(obj):
    while isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        obj = obj.item()
    return obj


def to_py(obj):
    obj = unwrap_mat(obj)
    if hasattr(obj, "_fieldnames"):
        return {fn: to_py(getattr(obj, fn)) for fn in obj._fieldnames}
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return np.vectorize(to_py, otypes=[object])(obj)
    return obj


def describe(name, x, max_vals=5):
    x = unwrap_mat(x)
    if hasattr(x, "_fieldnames"):
        print(f"{name}: MATLAB struct with fields -> {list(x._fieldnames)}")
    elif isinstance(x, np.ndarray):
        print(f"{name}: ndarray shape={x.shape}, dtype={x.dtype}")
        if x.size and x.ndim <= 2:
            flat = x.ravel()
            print(f"  first values: {flat[:max_vals]}")
    else:
        print(f"{name}: {type(x)} -> {x}")


def ensure_2d_array(x, name):
    x = unwrap_mat(x)
    if hasattr(x, "_fieldnames"):
        for k in ["EEG", "eeg", "data", "signal", "X"]:
            if hasattr(x, k):
                x = getattr(x, k)
                break
        else:
            raise ValueError(
                f"{name} parece struct pero no se encontró campo de datos.")
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(
            f"{name} debe ser 2D (channels x samples), shape={x.shape}")
    return x


def scale_pair(arr_con, arr_pure=None, kind="standard", per_channel=True,
               fit_on="both", feature_range=(-1, 1)):
    if arr_con.ndim != 2:
        raise ValueError("arr_con debe ser 2D (channels x samples).")
    if arr_pure is not None and arr_pure.ndim != 2:
        raise ValueError("arr_pure debe ser 2D (channels x samples).")
    if arr_pure is not None and arr_con.shape[0] != arr_pure.shape[0]:
        raise ValueError(
            f"Channel mismatch: con={arr_con.shape[0]} vs pure={arr_pure.shape[0]}")

    n_ch, _ = arr_con.shape
    arr_con_s = np.empty_like(arr_con, dtype=float)
    arr_pure_s = None if arr_pure is None else np.empty_like(
        arr_pure, dtype=float)
    scalers = []

    def make_scaler():
        if kind == "standard":
            return StandardScaler(with_mean=True, with_std=True)
        elif kind == "minmax":
            return MinMaxScaler(feature_range=feature_range)
        elif kind == "robust":
            return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
        else:
            raise ValueError("kind must be 'standard','minmax','robust'")

    if per_channel:
        for ch in range(n_ch):
            sc = make_scaler()
            if arr_pure is None:
                fit_vec = arr_con[ch]
            else:
                if fit_on == "pure":
                    fit_vec = arr_pure[ch]
                elif fit_on == "con":
                    fit_vec = arr_con[ch]
                elif fit_on == "both":
                    fit_vec = np.concatenate(
                        [arr_con[ch], arr_pure[ch]], axis=0)
                else:
                    raise ValueError("fit_on must be 'pure','con','both'")
            sc.fit(fit_vec.reshape(-1, 1))
            arr_con_s[ch] = sc.transform(arr_con[ch].reshape(-1, 1)).ravel()
            if arr_pure is not None:
                arr_pure_s[ch] = sc.transform(
                    arr_pure[ch].reshape(-1, 1)).ravel()
            scalers.append(sc)
    else:
        sc = make_scaler()
        if arr_pure is None:
            fit_mat = arr_con.T
        else:
            if fit_on == "pure":
                fit_mat = arr_pure.T
            elif fit_on == "con":
                fit_mat = arr_con.T
            elif fit_on == "both":
                fit_mat = np.vstack([arr_con.T, arr_pure.T])
            else:
                raise ValueError("fit_on must be 'pure','con','both'")
        sc.fit(fit_mat)
        arr_con_s = sc.transform(arr_con.T).T
        if arr_pure is not None:
            arr_pure_s = sc.transform(arr_pure.T).T
        scalers.append(sc)

    return arr_con_s, arr_pure_s, scalers


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


@torch.inference_mode()
def infer_ola_1d(
    model: nn.Module,
    x_1d: np.ndarray,
    device: torch.device,
    win: int = 512,
    hop: int = 512,
    batch_size: int = 256,
    use_autocast: bool = True,
    window_input: bool = False,
) -> np.ndarray:
    T = int(x_1d.shape[0])
    if T <= 0:
        return np.zeros_like(x_1d, dtype=np.float32)

    n_frames = max(1, int(np.ceil((T - win) / hop)) + 1) if T > win else 1
    out_len = (n_frames - 1) * hop + win
    pad_len = max(0, out_len - T)
    x_pad = np.pad(x_1d, (0, pad_len), mode="reflect")

    from numpy.lib.stride_tricks import sliding_window_view
    frames = sliding_window_view(x_pad, win)[::hop]
    assert frames.shape[0] == n_frames

    w = np.hanning(win).astype(np.float32)

    frames_in = frames if not window_input else (frames * w[None, :])

    outs = []
    bsz_list = [batch_size, 128, 64, 32, 16, 8, 4, 2, 1]

    def _clear():
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    for bsz in bsz_list:
        try:
            outs = []
            j = 0
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if (use_autocast and device.type == "cuda")
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                while j < n_frames:
                    sl = slice(j, min(j + bsz, n_frames))
                    xb_np = np.ascontiguousarray(
                        frames_in[sl], dtype=np.float32)
                    xb = torch.from_numpy(xb_np).unsqueeze(
                        1).to(device)
                    yb = model(xb).detach().cpu().squeeze(
                        1).numpy()
                    outs.append(yb)
                    j = sl.stop
                    _clear()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bsz != 1:
                print(f"[WARN] OOM batch_size={bsz}, probando menor…")
                continue
            else:
                raise

    y_frames = np.concatenate(outs, axis=0)
    y_rec = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop
        y_rec[start:start+win] += y_frames[i] * w
        wsum[start:start+win] += w

    wsum[wsum < 1e-8] = 1.0
    y_rec /= wsum

    return y_rec[:T]


@torch.inference_mode()
def infer_eeg_matrix(
    model: nn.Module,
    X_con_scaled: np.ndarray,
    device: torch.device,
    win: int, hop: int,
    batch_size: int,
    use_autocast: bool = True,
    window_input: bool = False,
) -> np.ndarray:
    C, T = X_con_scaled.shape
    yhat = np.zeros_like(X_con_scaled, dtype=np.float32)
    for ch in range(C):
        yhat[ch] = infer_ola_1d(
            model, X_con_scaled[ch], device,
            win=win, hop=hop, batch_size=batch_size,
            use_autocast=use_autocast, window_input=window_input
        )
    return yhat


def compute_metrics_pair(y_true_2d: np.ndarray, y_pred_2d: np.ndarray, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    assert y_true_2d.shape == y_pred_2d.shape
    y = y_true_2d
    p = y_pred_2d

    y_mean = y.mean(axis=1, keepdims=True)
    p_mean = p.mean(axis=1, keepdims=True)
    num = np.sum((y - y_mean) * (p - p_mean), axis=1)
    den = np.sqrt(np.sum((y - y_mean)**2, axis=1) *
                  np.sum((p - p_mean)**2, axis=1)) + eps
    cc = num / den

    err = p - y
    mse = np.mean(err**2, axis=1)
    rmse = np.sqrt(mse)
    rrmse = np.sqrt(np.sum(err**2, axis=1) / (np.sum(y**2, axis=1) + eps))

    return {"CC": cc, "MSE": mse, "RMSE": rmse, "RRMSE": rrmse}


def print_metrics_summary(title: str, met: Dict[str, np.ndarray]):
    print(f"\n===== {title} =====")
    for k, v in met.items():
        print(
            f"{k}: mean={v.mean():.4f} | std={v.std():.4f} (n={v.shape[0]} canales)")


def save_metrics_per_channel_csv(path: str, metrics: Dict[str, np.ndarray]):
    C = metrics["CC"].shape[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ch", "CC", "MSE", "RMSE", "RRMSE"])
        for ch in range(C):
            w.writerow([
                ch,
                float(metrics["CC"][ch]),
                float(metrics["MSE"][ch]),
                float(metrics["RMSE"][ch]),
                float(metrics["RRMSE"][ch]),
            ])
    print(f"[INFO] Métricas por canal -> {path}")


def save_metrics_summary_csv(path: str, den: Dict[str, np.ndarray], base: Dict[str, np.ndarray]):
    def mstats(d: Dict[str, np.ndarray]):
        return {k: (float(v.mean()), float(v.std())) for k, v in d.items()}

    ds = mstats(den)
    bs = mstats(base)
    rows = [
        ["set", "CC_mean", "CC_std", "MSE_mean", "MSE_std",
            "RMSE_mean", "RMSE_std", "RRMSE_mean", "RRMSE_std"],
        ["denoised", ds["CC"][0], ds["CC"][1], ds["MSE"][0], ds["MSE"][1],
            ds["RMSE"][0], ds["RMSE"][1], ds["RRMSE"][0], ds["RRMSE"][1]],
        ["baseline", bs["CC"][0], bs["CC"][1], bs["MSE"][0], bs["MSE"][1],
            bs["RMSE"][0], bs["RMSE"][1], bs["RRMSE"][0], bs["RRMSE"][1]],
        ["delta(den-baseline) CC", ds["CC"][0] -
         bs["CC"][0], "", "", "", "", "", "", ""],
        ["delta(baseline-den) MSE", bs["MSE"][0] -
         ds["MSE"][0], "", "", "", "", "", "", ""],
        ["delta(baseline-den) RMSE", bs["RMSE"][0] -
         ds["RMSE"][0], "", "", "", "", "", "", ""],
        ["delta(baseline-den) RRMSE", bs["RRMSE"][0] -
         ds["RRMSE"][0], "", "", "", "", "", "", ""],
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    print(f"[INFO] Resumen de métricas -> {path}")


def plot_triplet(con: np.ndarray, den: np.ndarray, pure: Optional[np.ndarray],
                 ch: int, start: int, end: int, title: str, save_path: Optional[str],
                 auto_ylim: bool = False, ylim_pct: float = 99.5):
    C, T = con.shape
    ch = max(0, min(ch, C - 1))
    start = max(0, start)
    end = T if end < 0 else min(end, T)
    if start >= end:
        raise ValueError(f"Rango inválido start={start} end={end}")
    t = np.arange(start, end)

    plt.figure(figsize=(12, 4))
    plt.plot(t, con[ch, start:end], label="contaminated", alpha=0.8)
    plt.plot(t, den[ch, start:end], label="denoised (inference)", alpha=0.9)
    if pure is not None:
        plt.plot(t, pure[ch, start:end], label="clean (target)", alpha=0.9)
        seg_met = compute_metrics_pair(
            pure[ch:ch+1, start:end], den[ch:ch+1, start:end]
        )
        cc = float(seg_met["CC"][0])
        rmse = float(seg_met["RMSE"][0])
        rrmse = float(seg_met["RRMSE"][0])
        title = f"{title} | CC={cc:.3f}, RMSE={rmse:.3f}, RRMSE={rrmse:.3f}"

    if auto_ylim:
        lo = np.percentile(den[ch, start:end], 100 - ylim_pct)
        hi = np.percentile(den[ch, start:end], ylim_pct)
        pad = 0.1 * (hi - lo + 1e-6)
        plt.ylim(lo - pad, hi + pad)

    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (scaled)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figura guardada -> {save_path}")
        plt.close()
    else:
        plt.show()


def parse_range(s: str) -> Tuple[int, int]:
    try:
        a, b = s.split(":")
        return int(a), int(b)
    except Exception:
        raise argparse.ArgumentTypeError("Formato rango START:END, ej. 0:5000")


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Inferir EEG desde .mat (cont/pure) con ResUNetTCN + OLA corregida + métricas + CSV.")
    ap.add_argument("--cont", default="Contaminated_Data.mat",
                    help="Ruta a Contaminated_Data.mat")
    ap.add_argument("--pure", default=None,
                    help="Ruta a Pure_Data.mat (opcional para métricas)")
    ap.add_argument("--sim-id", type=int, default=1,
                    help="ID para claves sim{ID}_con / sim{ID}_resampled")
    ap.add_argument("--cont-key", default=None,
                    help="Clave contaminada (default: sim{ID}_con)")
    ap.add_argument("--pure-key", default=None,
                    help="Clave limpia (default: sim{ID}_resampled)")
    ap.add_argument("--ckpt", default="best_joint_denoiser.pt",
                    help="Checkpoint del modelo (.pt)")
    ap.add_argument("--cpu", action="store_true", help="Forzar CPU")
    ap.add_argument("--batch-size", type=int,
                    default=256, help="Batch inicial")
    ap.add_argument("--no-autocast", action="store_true",
                    help="Desactiva autocast FP16 en CUDA")
    ap.add_argument("--win", type=int, default=512,
                    help="Tamaño de ventana del modelo")
    ap.add_argument("--hop", type=int, default=512,
                    help="Salto entre ventanas")
    ap.add_argument("--window-input", action="store_true",
                    help="Multiplicar también la ENTRADA por Hann (por defecto NO)")
    ap.add_argument(
        "--scaler", choices=["standard", "minmax", "robust"], default="standard")
    ap.add_argument(
        "--fit-on", choices=["both", "pure", "con"], default="both")
    ap.add_argument("--global-scaling", action="store_true",
                    help="Un solo scaler global (por defecto por canal)")
    ap.add_argument("--minmax-min", type=float, default=-1.0)
    ap.add_argument("--minmax-max", type=float, default=1.0)
    # Plot
    ap.add_argument("--ch", type=int, default=0, help="Canal a graficar")
    ap.add_argument("--seg", type=parse_range, default="0:5000",
                    help="Segmento START:END (samples), END<0 = todo")
    ap.add_argument("--auto-ylim", action="store_true",
                    help="Ajuste automático del eje Y (percentiles)")
    ap.add_argument("--ylim-pct", type=float, default=99.5,
                    help="Percentil superior para auto-ylim")
    ap.add_argument("--out-dir", default="./inferences",
                    help="Directorio de salida")
    ap.add_argument("--save-denoised-mat", action="store_true",
                    help="Guardar denoised en .mat")
    return ap


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu" if args.cpu else (
        "cuda" if torch.cuda.is_available() else "cpu"))
    use_autocast = (device.type == "cuda") and (not args.no_autocast)
    print(
        f"[INFO] Device={device} | autocast(fp16)={'ON' if use_autocast else 'OFF'}")
    base = os.path.splitext(os.path.basename(args.cont))[0]

    cont_dict = load_mat_as_dict(args.cont)
    cont_key = args.cont_key or f"sim{args.sim_id}_con"
    if cont_key not in cont_dict:
        raise KeyError(
            f"No se encontró clave '{cont_key}' en {args.cont}. Claves: {list(cont_dict.keys())}")
    sim_con_raw = cont_dict[cont_key]
    print("=== Objeto contaminado (raw) ===")
    describe(cont_key, sim_con_raw)
    sim_con = to_py(sim_con_raw)
    print("\n=== Después de conversión ===")
    describe(cont_key, sim_con)
    sim_con = ensure_2d_array(sim_con, cont_key)

    sim_pure = None
    if args.pure:
        pure_dict = load_mat_as_dict(args.pure)
        pure_key = args.pure_key or f"sim{args.sim_id}_resampled"
        if pure_key not in pure_dict:
            raise KeyError(
                f"No se encontró clave '{pure_key}' en {args.pure}. Claves: {list(pure_dict.keys())}")
        sim_pure_raw = pure_dict[pure_key]
        print("\n=== Objeto limpio (raw) ===")
        describe(pure_key, sim_pure_raw)
        sim_pure = to_py(sim_pure_raw)
        print("\n=== Después de conversión ===")
        describe(pure_key, sim_pure)
        sim_pure = ensure_2d_array(sim_pure, pure_key)
        if sim_pure.shape[0] != sim_con.shape[0]:
            raise ValueError(
                f"Channel count difiere: con={sim_con.shape} vs pure={sim_pure.shape}")
        if sim_pure.shape[1] != sim_con.shape[1]:
            print(
                f"[WARN] Longitudes distintas: con={sim_con.shape} vs pure={sim_pure.shape}. Se recortará al mínimo.")
            T = min(sim_con.shape[1], sim_pure.shape[1])
            sim_con = sim_con[:, :T]
            sim_pure = sim_pure[:, :T]

    print("\n=== Config de escalado ===")
    print(f"scaler={args.scaler}, fit_on={args.fit_on}, per_channel={not args.global_scaling}, "
          f"feature_range=({args.minmax_min},{args.minmax_max})")
    sim_con_s, sim_pure_s, _ = scale_pair(
        sim_con, sim_pure,
        kind=args.scaler,
        per_channel=(not args.global_scaling),
        fit_on=args.fit_on,
        feature_range=(args.minmax_min, args.minmax_max)
    )

    print(f"\n[INFO] Cargando modelo desde: {args.ckpt}")
    model = load_model(args.ckpt, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"\n[INFO] Inferencia OLA: win={args.win}, hop={args.hop}, batch={args.batch_size}, window_input={args.window_input}")
    yhat_s = infer_eeg_matrix(
        model=model,
        X_con_scaled=sim_con_s.astype(np.float32),
        device=device,
        win=args.win,
        hop=args.hop,
        batch_size=args.batch_size,
        use_autocast=use_autocast,
        window_input=args.window_input,
    )  # (C, T)

    npy_path = os.path.join(args.out_dir, f"{base}_denoised_scaled.npy")
    np.save(npy_path, yhat_s)
    print(f"[INFO] Denoised (scaled) guardado -> {npy_path}")

    if args.save_denoised_mat:
        mat_path = os.path.join(args.out_dir, f"{base}_denoised_scaled.mat")
        savemat(mat_path, {"denoised_scaled": yhat_s})
        print(f"[INFO] Denoised (scaled) guardado en .mat -> {mat_path}")

    if sim_pure_s is not None:
        den_metrics = compute_metrics_pair(
            sim_pure_s.astype(np.float32), yhat_s.astype(np.float32))
        base_metrics = compute_metrics_pair(
            sim_pure_s.astype(np.float32), sim_con_s.astype(np.float32))

        print_metrics_summary("Métricas (DENOISED vs CLEAN)", den_metrics)
        print_metrics_summary(
            "Métricas (NOISY vs CLEAN) [baseline]", base_metrics)

        den_csv = os.path.join(
            args.out_dir, f"metrics_per_channel_{base}_denoised.csv")
        base_csv = os.path.join(
            args.out_dir, f"metrics_per_channel_{base}_baseline.csv")
        save_metrics_per_channel_csv(den_csv, den_metrics)
        save_metrics_per_channel_csv(base_csv, base_metrics)

        sum_csv = os.path.join(args.out_dir, f"metrics_summary_{base}.csv")
        save_metrics_summary_csv(sum_csv, den_metrics, base_metrics)

    else:
        print(
            "\n[INFO] No se proporcionó 'Pure_Data.mat'. Se omiten métricas vs. limpio y CSVs.")

    s_start, s_end = args.seg
    title = f"Denoising | ch {args.ch} | seg {s_start}:{s_end if s_end >= 0 else 'end'}"
    fig_path = os.path.join(args.out_dir, f"{base}_triplet.png")
    plot_triplet(
        sim_con_s, yhat_s, sim_pure_s,
        ch=args.ch, start=s_start, end=s_end,
        title=title, save_path=fig_path,
        auto_ylim=args.auto_ylim, ylim_pct=args.ylim_pct
    )

    print("\n[OK] Proceso completado.")


if __name__ == "__main__":
    main()
