import os
import gc
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple

from utils.model import ResUNetTCN


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    s = float(n)
    for u in units:
        if s < 1024:
            return f"{s:.1f}{u}"
        s /= 1024
    return f"{s:.1f}TB"


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


def compute_hop_len(win_len: int, overlap: float) -> int:
    overlap = float(overlap)
    overlap = max(0.0, min(0.95, overlap))
    hop = int(round(win_len * (1.0 - overlap)))
    return max(1, hop)


def frame_signal(x: np.ndarray, win_len: int, hop_len: int) -> Tuple[np.ndarray, int]:
    L = int(x.shape[0])
    if L <= 0:
        raise ValueError("La señal de entrada está vacía")
    if win_len <= 0:
        raise ValueError("win_len debe ser > 0")

    if L <= win_len:
        pad_len = win_len - L
        x_pad = np.pad(x, (0, pad_len), mode="constant")
        return x_pad[None, :], x_pad.shape[0]

    n_frames = 1 + math.ceil((L - win_len) / hop_len)
    pad_needed = max(0, (n_frames - 1) * hop_len + win_len - L)
    x_pad = np.pad(x, (0, pad_needed), mode="constant")
    Lpad = x_pad.shape[0]

    frames = []
    start = 0
    for _ in range(n_frames):
        frames.append(x_pad[start:start + win_len])
        start += hop_len
    X = np.stack(frames, axis=0)
    return X, Lpad


def ola_reconstruct(frames: np.ndarray, L_out: int, hop_len: int) -> np.ndarray:
    n_frames, win_len = frames.shape
    out = np.zeros((L_out,), dtype=frames.dtype)
    wsum = np.zeros((L_out,), dtype=frames.dtype)
    t = 0
    for i in range(n_frames):
        seg_end = t + win_len
        out[t:seg_end] += frames[i]
        wsum[t:seg_end] += 1.0
        t += hop_len
    wsum[wsum == 0] = 1.0
    out = out / wsum
    return out


@torch.inference_mode()
def infer_windows_batched(model: nn.Module,
                          windows: np.ndarray,
                          device: torch.device,
                          start_bs: int = 256,
                          use_autocast: bool = True) -> np.ndarray:
    N, L = windows.shape
    bsz_candidates = [start_bs, 128, 64, 32, 16, 8, 4, 2, 1]
    out_chunks = []

    def _clear():
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    for bsz in bsz_candidates:
        try:
            _clear()
            out_chunks = []
            j = 0
            if device.type == "cuda" and use_autocast:
                ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
            else:
                class Dummy:
                    def __enter__(self):
                        return None

                    def __exit__(self, *args):
                        return False
                ctx = Dummy()
            with ctx:
                while j < N:
                    xb = windows[j:j+bsz]
                    xb = torch.from_numpy(xb).float()
                    xb = xb.unsqueeze(1)
                    xb = xb.to(device, non_blocking=True)
                    yb = model(xb)
                    yb = yb.detach().float().cpu().squeeze(1).numpy()
                    out_chunks.append(yb)
                    del xb
                    j += bsz
                    _clear()
            print(f"[INFO] Inference OK with batch_size={bsz}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARN] OOM at batch_size={bsz}, trying smaller…")
                _clear()
                continue
            raise

    yhat = np.concatenate(out_chunks, axis=0)
    return yhat


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    num = float(np.sum(ref ** 2)) + 1e-12
    den = float(np.sum((ref - est) ** 2)) + 1e-12
    return 10.0 * math.log10(num / den)


def mse(ref: np.ndarray, est: np.ndarray) -> float:
    return float(np.mean((ref - est) ** 2))


def corrcoef(ref: np.ndarray, est: np.ndarray) -> float:
    r = np.corrcoef(ref, est)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(r)


def parse_range(s: str) -> Tuple[int, int]:
    a, b = s.split(":")
    return int(a), int(b)


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia 1D con ResUNetTCN sobre archivos .npy concatenados.")

    default_noisy = os.path.join(
        "semi-simulated-EEGEOG-dataset", "salida_con_1d.npy")
    default_clean = os.path.join(
        "semi-simulated-EEGEOG-dataset", "salida_pure_1d.npy")

    parser.add_argument("--noisy-path", type=str, default=default_noisy,
                        help="Ruta al .npy de señal contaminada (1D)")
    parser.add_argument("--clean-path", type=str, default=default_clean,
                        help="Ruta al .npy de señal limpia (1D)")
    parser.add_argument("--ckpt", type=str, default="best_joint_denoiser.pt",
                        help="Checkpoint del modelo")
    parser.add_argument("--out-dir", type=str, default="inferences",
                        help="Carpeta de salida")

    parser.add_argument("--win-len", type=int, default=2048,
                        help="Tamaño de ventana (L)")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="Solapamiento entre 0.0 y 0.95 (0=sin solape)")
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size inicial para intentar")
    parser.add_argument("--no-amp", action="store_true",
                        help="Desactiva autocast/AMP en GPU")

    parser.add_argument("--preview-win", type=int, default=512,
                        help="Tamaño de cada preview (muestras)")
    parser.add_argument("--preview-hop", type=int, default=512,
                        help="Paso entre previews (muestras)")
    parser.add_argument("--limit-previews", type=int, default=None,
                        help="Máximo de previews a generar (para acelerar pruebas)")

    parser.add_argument("--plot-seg", type=parse_range, default=None,
                        help="Segmento único START:END para graficar además de los previews por chunks")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = (device.type == "cuda") and (not args.no_amp)

    set_seed(42)

    model = load_model(args.ckpt, device)
    print(f"[INFO] Modelo cargado de: {args.ckpt}")

    noisy = np.load(args.noisy_path)
    clean = np.load(args.clean_path)
    if noisy.ndim != 1:
        noisy = noisy.reshape(-1)
    if clean.ndim != 1:
        clean = clean.reshape(-1)

    L = min(len(noisy), len(clean))
    noisy = noisy[:L].astype(np.float32)
    clean = clean[:L].astype(np.float32)
    print(
        f"[INFO] Señal cargada | long = {L:,} muestras | noisy: {human_size(noisy.nbytes)} | clean: {human_size(clean.nbytes)}")

    hop = compute_hop_len(args.win_len, args.overlap)
    X_win, Lpad = frame_signal(noisy, args.win_len, hop)
    Y_win, _ = frame_signal(clean, args.win_len, hop)
    print(
        f"[INFO] Ventanas -> N={X_win.shape[0]} | win_len={args.win_len} | hop={hop} | Lpad={Lpad}")

    yhat_win = infer_windows_batched(model, X_win, device,
                                     start_bs=args.batch, use_autocast=use_autocast)

    denoised = ola_reconstruct(yhat_win, Lpad, hop)[:L].astype(np.float32)

    snr_in = snr_db(clean, noisy)
    snr_out = snr_db(clean, denoised)
    imp = snr_out - snr_in
    mse_in = mse(clean, noisy)
    mse_out = mse(clean, denoised)
    r_out = corrcoef(clean, denoised)

    metrics = {
        "win_len": args.win_len,
        "overlap": args.overlap,
        "batch": args.batch,
        "device": str(device),
        "snr_in_dB": snr_in,
        "snr_out_dB": snr_out,
        "snr_improvement_dB": imp,
        "mse_in": mse_in,
        "mse_out": mse_out,
        "corr_out": r_out,
        "N_frames": int(X_win.shape[0]),
        "L_signal": int(L),
    }

    den_path = os.path.join(args.out_dir, "denoised_1d.npy")
    np.save(den_path, denoised)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pw = max(1, int(args.preview_win))
    ph = max(1, int(args.preview_hop))
    n_gen = 0
    for s in range(0, L, ph):
        e = min(s + pw, L)
        if e - s <= 1:
            break
        fig = plt.figure(figsize=(12, 4))
        t = np.arange(s, e)
        plt.plot(t, noisy[s:e], label="noisy", alpha=0.6)
        plt.plot(t, denoised[s:e], label="denoised", alpha=0.8)
        plt.plot(t, clean[s:e], label="clean", alpha=0.8)
        plt.title(
            f"Preview {s}:{e} | SNR_in={snr_in:.2f} dB | SNR_out={snr_out:.2f} dB")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend(loc="best")
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, f"preview_{s}_{e}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        n_gen += 1
        if args.limit_previews is not None and n_gen >= int(args.limit_previews):
            break

    if args.plot_seg is not None:
        s, e = args.plot_seg
        s = max(0, min(s, L-1))
        e = max(s+1, min(e, L))
        fig = plt.figure(figsize=(12, 4))
        t = np.arange(s, e)
        plt.plot(t, noisy[s:e], label="noisy", alpha=0.6)
        plt.plot(t, denoised[s:e], label="denoised", alpha=0.8)
        plt.plot(t, clean[s:e], label="clean", alpha=0.8)
        plt.title(
            f"Segment {s}:{e} | SNR_in={snr_in:.2f} dB | SNR_out={snr_out:.2f} dB")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend(loc="best")
        plt.tight_layout()
        fig_path_single = os.path.join(
            args.out_dir, f"preview_single_{s}_{e}.png")
        plt.savefig(fig_path_single, dpi=150)
        plt.close(fig)
        print(f"  → Preview (único) : {fig_path_single}")

    print("\n[OK] Inference finished.")
    print(f"  → Denoised: {den_path}")
    print(f"  → Metrics : {os.path.join(args.out_dir, 'metrics.json')}")
    print(
        f"  → Previews: {n_gen} archivos en {args.out_dir} (tamaño {pw} y salto {ph})")
    print(f"  SNR in/out (dB): {snr_in:.2f} → {snr_out:.2f} (Δ {imp:.2f} dB)")


if __name__ == "__main__":
    main()


# python infer.py ^
#   --noisy-path "semi-simulated-EEGEOG-dataset\salida_con_1d.npy" ^
#   --clean-path "semi-simulated-EEGEOG-dataset\salida_pure_1d.npy" ^
#   --ckpt "best_joint_denoiser.pt" ^
#   --out-dir "inferences" ^
#   --win-len 512 ^
#   --overlap 1.0
