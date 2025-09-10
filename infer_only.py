import os
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from utils.model import ResUNetTCN

MODEL_CKPT = "./best_joint_denoiser.pt"
OUT_DIR = "./inferences"
NOISES = ["EMG", "EOG", "CHEW", "SHIV"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AUTOMIX = torch.cuda.is_available()
DEFAULT_BS = 256
torch.manual_seed(42)
np.random.seed(42)
os.makedirs(OUT_DIR, exist_ok=True)


def load_model(ckpt_path: str, device: torch.device):
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


def load_test_tensors(noise: str):
    X = np.load(f"./data/data_for_test/X_test_{noise}.npy")
    y = np.load(f"./data/data_for_test/y_test_{noise}.npy")
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    return X, y


@torch.inference_mode()
def infer_batched(
    model: nn.Module,
    X_test: torch.Tensor,
    device: torch.device,
    start_bs: int = DEFAULT_BS,
    use_autocast: bool = USE_AUTOMIX
) -> torch.Tensor:
    S, M, C, L = X_test.shape
    N = S * M
    flat = X_test.reshape(N, C, L)

    bsz_candidates = [start_bs, 128, 64, 32, 16, 8, 4, 2, 1]
    out_cpu = []
    i = 0

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
                torch.autocast(device_type="cuda", dtype=torch.float16) if (use_autocast and device.type == "cuda")
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
            print(f"[INFO] Inference OK with batch_size={bsz}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARN] OOM at batch_size={bsz}, trying smaller…")
                _clear()
                continue
            else:
                raise

    yhat = torch.cat(out_cpu, dim=0).reshape(S, M, C, L)
    return yhat


def animate_across_snr(
    noise_name: str,
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    sample_idx: int = 0,
    out_path: str | None = None,
    fps: int = 2,
    device: torch.device = DEVICE,
) -> str:
    model.eval()
    S, M, C, L = X_test.shape
    assert C == 1, "Expect 1 channel"
    t = np.arange(L)

    noisy_np = X_test[:, sample_idx, 0, :].numpy()
    clean_np = y_test[:, sample_idx, 0, :].numpy()
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
    ax.set_xlim(0, L - 1)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"{noise_name} | sample {sample_idx} | across SNR")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")

    def init():
        ln_noisy.set_data([], [])
        ln_pred.set_data([], [])
        ln_clean.set_data([], [])
        return ln_noisy, ln_pred, ln_clean

    @torch.inference_mode()
    def update(frame):
        xb_cpu = X_test[frame, sample_idx].unsqueeze(0)
        yb_cpu = y_test[frame, sample_idx].unsqueeze(0)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yhat = model(xb_cpu.to(device)).cpu()
        else:
            yhat = model(xb_cpu)

        ln_noisy.set_data(t, xb_cpu.squeeze(0).squeeze(0).numpy())
        ln_pred.set_data(t, yhat.squeeze(0).squeeze(0).numpy())
        ln_clean.set_data(t, yb_cpu.squeeze(0).squeeze(0).numpy())
        ax.set_xlabel(f"Time (samples) | SNR idx = {frame}")
        return ln_noisy, ln_pred, ln_clean

    anim = FuncAnimation(fig, update, frames=S, init_func=init, blit=True)
    out_path = out_path or os.path.join(
        OUT_DIR, f"infer_{noise_name}_sample{sample_idx}.gif")
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    model = load_model(MODEL_CKPT, DEVICE)
    print("[INFO] Model loaded from:", MODEL_CKPT)

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    for noise in NOISES:
        print(f"\n[INFO] Inference on TEST ONLY — {noise}")
        X_test, y_test = load_test_tensors(noise)
        S, M, C, L = X_test.shape
        print(
            f"  X_test shape: {tuple(X_test.shape)} | y_test shape: {tuple(y_test.shape)}")

        yhat_test = infer_batched(
            model, X_test, DEVICE, start_bs=DEFAULT_BS, use_autocast=USE_AUTOMIX)

        pred_out = os.path.join(OUT_DIR, f"pred_{noise}.pt")
        torch.save({"yhat": yhat_test, "X_test": X_test,
                   "y_test": y_test}, pred_out)
        print(f"  Saved predictions -> {pred_out}")

        gif_path = animate_across_snr(
            noise_name=noise,
            model=model,
            X_test=X_test,
            y_test=y_test,
            sample_idx=0,
            out_path=None,
            fps=2,
            device=DEVICE,
        )
        print(f"  Saved animation -> {gif_path}")

    print("\n[OK] Inference completed for all requested noise types.")
