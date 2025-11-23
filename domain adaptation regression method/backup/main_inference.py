#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Inference for EEG/EMG denoising (PyTorch) using semi-simulated EEG/EOG .mat inputs.

Inputs (defaults preset):
    semi-simulated-EEGEOG-dataset/Contaminated_Data.mat
    semi-simulated-EEGEOG-dataset/Pure_Data.mat

Model formats:
  A) TorchScript (torch.jit.load)
  B) Full nn.Module saved with torch.save(model, ...)
  C) state_dict saved with torch.save(model.state_dict(), ...),
     in which case provide --model-init utils.model:ResUNetTCN
     and (optionally) --model-init-kwargs as JSON.

Outputs:
  - predictions: <out_dir>/preds_EMG.npy
  - metrics:     <out_dir>/metrics.txt
  - plots:       <out_dir>/plots/example_*.png
"""

import argparse
import importlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ---------------- Data helpers ---------------- #

def _load_mat_vars(path):
    """Load a .mat file as a dict, skip __* keys."""
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def _parse_range_list(s):
    """'all' | '0,3,5' | '1-4' | '1-4,7,10-12' -> sorted unique list[int] or None for 'all'."""
    if s is None:
        return None
    s = str(s).strip().lower()
    if s == "all":
        return None
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            for v in range(min(a, b), max(a, b) + 1):
                out.add(v)
        else:
            out.add(int(part))
    return sorted(out)


def _normalize_1xL_list(arrs, mode="min", fixed_len=None, pad_value=0.0):
    """
    Make all [1, L] arrays the same length.
    mode: 'min' (crop to shortest), 'max' (pad to longest), 'fixed' (pad/crop to fixed_len).
    Returns (normalized_list, used_length).
    """
    lens = [a.shape[-1] for a in arrs]
    if mode == "min":
        Lout = int(min(lens))
    elif mode == "max":
        Lout = int(max(lens))
    elif mode == "fixed":
        if fixed_len is None:
            raise ValueError("mode='fixed' requires --fixed_len")
        Lout = int(fixed_len)
    else:
        raise ValueError("length_mode must be 'min', 'max', or 'fixed'")

    norm = []
    for a in arrs:
        a = np.asarray(a, dtype=np.float32)
        L = a.shape[-1]
        if L == Lout:
            norm.append(a)
        elif L > Lout:
            norm.append(a[..., :Lout])  # crop right
        else:
            pad = Lout - L
            norm.append(np.pad(a, ((0, 0), (0, pad)),
                        mode="constant", constant_values=pad_value))
    return norm, Lout


def load_pair_from_mat(cont_path, pure_path, sims="all", channels="all",
                       length_mode="min", fixed_len=None, pad_value=0.0, verbose=True):
    """
    Load .mat pair where contaminated has keys sim{i}_con and pure has sim{i}_resampled.
    Return X, Y in shape [N, 1, L] (each channel becomes one sample), plus meta.
    Handles variable lengths across simulations via length_mode.
    """
    cont_path = Path(cont_path)
    pure_path = Path(pure_path)
    if not cont_path.exists():
        raise FileNotFoundError(f"Missing file: {cont_path}")
    if not pure_path.exists():
        raise FileNotFoundError(f"Missing file: {pure_path}")

    cont = _load_mat_vars(str(cont_path))
    pure = _load_mat_vars(str(pure_path))

    # discover simulation indices from contaminated keys
    sim_ids = []
    for k in cont.keys():
        if k.startswith("sim") and k.endswith("_con"):
            try:
                sim_ids.append(int(k[3:].split("_")[0]))
            except Exception:
                pass
    sim_ids = sorted(sim_ids)
    if not sim_ids:
        raise RuntimeError(
            "No 'sim*_con' variables found in contaminated .mat")

    sim_filter = _parse_range_list(sims)
    if sim_filter is not None:
        sim_ids = [i for i in sim_ids if i in sim_filter]
    if not sim_ids:
        raise RuntimeError("Simulation filter produced empty set.")

    X_list_raw, Y_list_raw = [], []
    lengths_seen = set()

    for i in sim_ids:
        ck = f"sim{i}_con"
        pk = f"sim{i}_resampled"
        if ck not in cont or pk not in pure:
            raise KeyError(f"Missing pair: {ck} or {pk}")

        X2 = np.asarray(cont[ck], dtype=np.float32)  # [C, L]
        Y2 = np.asarray(pure[pk], dtype=np.float32)  # [C, L]
        if X2.ndim != 2 or Y2.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays [C, L] per sim, got {X2.shape} / {Y2.shape} at sim{i}")
        if X2.shape[0] != Y2.shape[0]:
            raise ValueError(
                f"Channel count mismatch at sim{i}: {X2.shape[0]} vs {Y2.shape[0]}")
        if X2.shape[1] != Y2.shape[1]:
            # same sim pair should match; if not, crop to min for this sim
            Lmin = min(X2.shape[1], Y2.shape[1])
            X2, Y2 = X2[:, :Lmin], Y2[:, :Lmin]

        C, L = X2.shape
        lengths_seen.add(L)

        ch_filter = _parse_range_list(channels)
        if ch_filter is None:
            ch_indices = range(C)
        else:
            ch_indices = [ch for ch in ch_filter if 0 <= ch < C]
            if not ch_indices:
                raise RuntimeError(
                    f"Channel filter produced empty set for sim{i} (C={C}).")

        for ch in ch_indices:
            X_list_raw.append(X2[ch][None, :])  # [1, L]
            Y_list_raw.append(Y2[ch][None, :])  # [1, L]

    if verbose and len(lengths_seen) > 1:
        print(
            f"[WARN] Multiple lengths detected across sims: {sorted(lengths_seen)}")
        print(f"[INFO] Applying length_mode='{length_mode}'" +
              (f" with fixed_len={fixed_len}" if length_mode == "fixed" else "") +
              f" and pad_value={pad_value}")

    # unify lengths across all samples
    X_norm, Lout = _normalize_1xL_list(X_list_raw, mode=length_mode,
                                       fixed_len=fixed_len, pad_value=pad_value)
    Y_norm, _ = _normalize_1xL_list(Y_list_raw, mode=length_mode,
                                    fixed_len=Lout, pad_value=pad_value)

    X_arr = np.stack(X_norm, axis=0).astype(np.float32)  # [N, 1, Lout]
    Y_arr = np.stack(Y_norm, axis=0).astype(np.float32)  # [N, 1, Lout]
    meta = {"layout": "n_1_l", "N": X_arr.shape[0], "L": X_arr.shape[2]}
    return X_arr, Y_arr, meta


def load_npy_pair(x_path, y_path):
    X = np.load(x_path, allow_pickle=False)
    Y = np.load(y_path, allow_pickle=False)
    assert X.shape == Y.shape, f"X and Y shapes must match, got {X.shape} vs {Y.shape}"

    if X.ndim == 4:  # [SNR, M, 1, L]
        S, M, C, L = X.shape
        assert C == 1, f"Expected single channel; got C={C}"
        Xf = X.reshape(S * M, C, L)
        Yf = Y.reshape(S * M, C, L)
        meta = {"layout": "snr_m_1_l", "S": S, "M": M, "L": L}
    elif X.ndim == 3:  # [N, 1, L]
        N, C, L = X.shape
        assert C == 1, f"Expected single channel; got C={C}"
        Xf, Yf = X, Y
        meta = {"layout": "n_1_l", "N": N, "L": L}
    elif X.ndim == 2:  # [N, L] -> add channel
        N, L = X.shape
        Xf = X[:, None, :]
        Yf = Y[:, None, :]
        meta = {"layout": "n_l", "N": N, "L": L}
    else:
        raise ValueError(f"Unexpected ndim={X.ndim}. Supported: 2D, 3D, 4D.")
    return Xf.astype(np.float32), Yf.astype(np.float32), meta


# ---------------- Model loading ---------------- #

def _build_from_spec(factory_spec: str, kwargs: dict):
    """
    factory_spec: 'package.module:ClassOrFunction'
    kwargs: dict of constructor args (parsed from JSON)
    """
    if ":" not in factory_spec:
        raise ValueError("Use --model-init like 'utils.model:ResUNetTCN'")
    module_name, obj_name = factory_spec.split(":", 1)
    mod = importlib.import_module(module_name)
    if not hasattr(mod, obj_name):
        raise AttributeError(f"'{module_name}' has no attribute '{obj_name}'")
    factory = getattr(mod, obj_name)
    return factory(**(kwargs or {}))


def try_load_model(model_path, device, model_init=None, model_init_kwargs=None):
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}")

    # A) TorchScript
    try:
        model = torch.jit.load(str(mp), map_location=device)
        model.eval()
        return model
    except Exception:
        pass

    # B/C) torch.load
    obj = torch.load(str(mp), map_location=device)

    # Full nn.Module
    if hasattr(obj, "eval"):
        obj.eval()
        return obj

    # state_dict
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        if model_init is None:
            from utils.model import ResUNetTCN  # your default model class
            model = ResUNetTCN()
        else:
            model = _build_from_spec(model_init, model_init_kwargs)
        missing, unexpected = model.load_state_dict(obj, strict=False)
        if missing:
            print("[WARN] Missing keys when loading state_dict:", missing)
        if unexpected:
            print("[WARN] Unexpected keys when loading state_dict:", unexpected)
        model.to(device).eval()
        return model

    raise RuntimeError(
        "File is not TorchScript or a full nn.Module, and not a state_dict either."
    )


# ---------------- Inference & metrics ---------------- #

def batched_predict(model, X, device, batch_size=256):
    N = X.shape[0]
    preds = np.empty_like(X, dtype=np.float32)
    model.to(device)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            xb = torch.from_numpy(X[s:e]).to(device)
            yb = model(xb)
            yb = yb.detach().cpu().numpy().astype(np.float32)
            if yb.ndim == 2:
                yb = yb[:, None, :]
            preds[s:e] = yb
    return preds


def mse(a, b): return float(np.mean((a - b) ** 2))
def mae(a, b): return float(np.mean(np.abs(a - b)))


def snr_db(signal, noise):
    eps = 1e-12
    s_pow = np.sum(signal ** 2, axis=(-1, -2)) + eps
    n_pow = np.sum(noise ** 2, axis=(-1, -2)) + eps
    snr = 10.0 * np.log10(s_pow / n_pow)
    return float(np.mean(snr))


def compute_metrics(X_noisy, Y_true, Y_pred):
    in_noise = X_noisy - Y_true
    out_noise = Y_pred - Y_true
    return {
        "mse_pred_vs_true": mse(Y_pred, Y_true),
        "mae_pred_vs_true": mae(Y_pred, Y_true),
        "snr_input_db": snr_db(Y_true, in_noise),
        "snr_output_db": snr_db(Y_true, out_noise),
        "snr_improvement_db": snr_db(Y_true, out_noise) - snr_db(Y_true, in_noise),
    }


def plot_examples(X, Y_true, Y_pred, L, save_dir, num=6, snr_index=None, meta=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    N = X.shape[0]
    idxs = np.linspace(0, N - 1, num=min(num, N), dtype=int)

    if meta and meta.get("layout") == "snr_m_1_l" and snr_index is not None and meta.get("S", 0) > 0:
        S, M = meta["S"], meta["M"]
        si = int(snr_index) % S
        start, end = si * M, si * M + M
        idxs = np.linspace(start, end - 1, num=min(num, M), dtype=int)

    x_ax = np.arange(L)
    for k, i in enumerate(idxs, 1):
        fig = plt.figure(figsize=(10, 5))
        plt.title(f"Sample {i}")
        plt.plot(x_ax, X[i, 0], label="X (noisy)")
        plt.plot(x_ax, Y_pred[i, 0], label="Prediction")
        plt.plot(x_ax, Y_true[i, 0], label="Y (clean)")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend(loc="best")
        plt.tight_layout()
        fig.savefig(save_dir / f"example_{k:02d}_idx{i}.png", dpi=150)
        plt.close(fig)


# ---------------- CLI ---------------- #

def parse_args():
    ap = argparse.ArgumentParser(
        description="Run inference for denoising and plot results (supports .mat or .npy inputs).")
    ap.add_argument("--model", type=str, required=True,
                    help="Path to model file (.pt/.pth).")
    ap.add_argument("--model-init", type=str, default=None,
                    help="Factory for state_dict, e.g. utils.model:ResUNetTCN")
    ap.add_argument("--model-init-kwargs", type=str, default=None,
                    help='JSON dict of kwargs, e.g. {"in_ch":1,"base":64,"depth":3,"k":7}')

    # MAT inputs (defaults set to your paths)
    ap.add_argument("--mat_cont", type=str,
                    default="semi-simulated-EEGEOG-dataset/Contaminated_Data.mat",
                    help="Path to Contaminated_Data.mat")
    ap.add_argument("--mat_pure", type=str,
                    default="semi-simulated-EEGEOG-dataset/Pure_Data.mat",
                    help="Path to Pure_Data.mat")
    ap.add_argument("--mat_sims", type=str, default="all",
                    help="Which simulations to use, e.g., 'all' or '1-54' or '1-10,20'")
    ap.add_argument("--mat_channels", type=str, default="all",
                    help="Which channels to use, e.g., 'all' or '0-18' or '0,3,5'")

    # Length normalization across sims
    ap.add_argument("--length_mode", type=str, default="min",
                    choices=["min", "max", "fixed"],
                    help="How to make all samples the same length: crop to min, pad to max, or fixed length.")
    ap.add_argument("--fixed_len", type=int, default=None,
                    help="Used when --length_mode fixed; pad/crop to this length.")
    ap.add_argument("--pad_value", type=float, default=0.0,
                    help="Pad value when padding is needed.")

    # NPY fallback (kept for compatibility)
    ap.add_argument("--x_path", type=str,
                    default="data/data_for_test/X_test_EOG.npy")
    ap.add_argument("--y_path", type=str,
                    default="data/data_for_test/y_test_EOG.npy")

    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--plots", type=int, default=6)
    ap.add_argument("--snr_index", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default="outputs")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Prefer MAT inputs (your case)
    use_mat = Path(args.mat_cont).exists() and Path(args.mat_pure).exists()
    if use_mat:
        X, Y, meta = load_pair_from_mat(
            cont_path=args.mat_cont,
            pure_path=args.mat_pure,
            sims=args.mat_sims,
            channels=args.mat_channels,
            length_mode=args.length_mode,
            fixed_len=args.fixed_len,
            pad_value=args.pad_value,
            verbose=True
        )
    else:
        # Fallback to provided npy paths
        X, Y, meta = load_npy_pair(args.x_path, args.y_path)

    N, C, L = X.shape
    print(f"[INFO] Loaded data X,Y with shape {X.shape}, meta={meta}")

    # Build/load model
    kwargs = json.loads(
        args.model_init_kwargs) if args.model_init_kwargs else None
    model = try_load_model(args.model, device=device,
                           model_init=args.model_init, model_init_kwargs=kwargs)
    print("[INFO] Model loaded; running inference...")

    # Predict
    preds = batched_predict(model, X, device=device,
                            batch_size=args.batch_size)
    np.save(out_dir / "preds_EMG.npy", preds)
    print(f"[INFO] Saved predictions to {out_dir/'preds_EMG.npy'}")

    # Metrics
    metrics = compute_metrics(X_noisy=X, Y_true=Y, Y_pred=preds)
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    # Plots
    plot_examples(X, Y, preds, L=L, save_dir=out_dir / "plots",
                  num=args.plots, snr_index=args.snr_index, meta=meta)
    print(f"[INFO] Plots saved to {out_dir/'plots'}")


if __name__ == "__main__":
    main()
