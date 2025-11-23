# denoise_and_plot.py
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =========================
#   Utils: seguridad de carga y reshapes
# =========================


def _collect_numpy_safe_globals():
    base = [
        np.ndarray, np.dtype, np.generic, np.number, np.bool_,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
        np.complex64, np.complex128,
    ]
    try:
        base.append(np.core.multiarray._reconstruct)
    except Exception:
        pass
    try:
        dtypes_mod = getattr(np, "dtypes", None)
        if dtypes_mod is not None:
            for name in dir(dtypes_mod):
                if name.endswith("DType"):
                    cls = getattr(dtypes_mod, name)
                    if isinstance(cls, type):
                        base.append(cls)
    except Exception:
        pass
    return base


_SAFE_NUMPY_GLOBALS = _collect_numpy_safe_globals()
try:
    torch.serialization.add_safe_globals(_SAFE_NUMPY_GLOBALS)
except Exception:
    pass


def _reshape_segments_to_N1L(obj, L_expected=None):
    """
    Devuelve tensor [N,1,L]. Acepta tensor, array, lista o dict con claves comunes.
    Si L_expected es None, infiere L de la última dimensión.
    """
    if isinstance(obj, dict):
        for k in ("data", "signal", "x", "X"):
            if k in obj:
                obj = obj[k]
                break
    if isinstance(obj, np.ndarray):
        t = torch.from_numpy(obj.astype(np.float32, copy=False))
    else:
        t = torch.as_tensor(obj, dtype=torch.float32)
    t = t.detach().cpu()
    if t.ndim == 1:
        L = t.numel()
        if L_expected is not None and L != L_expected:
            raise ValueError(f"Segmento de longitud {L} != {L_expected}")
        return t.view(1, 1, L)
    # [N, L] o [*,*,L]
    L = t.shape[-1]
    if L_expected is not None and L != L_expected:
        # Caso especial: [L, ...] -> permutar
        if t.shape[0] == L_expected and t.ndim >= 2:
            t = t.permute(*range(1, t.ndim), 0).contiguous()
            L = t.shape[-1]
            if L != L_expected:
                raise ValueError(
                    f"No pude conformar [*,*,{L_expected}] desde {tuple(t.shape)}")
        else:
            raise ValueError(f"Longitud {L} != {L_expected}")
    return t.reshape(-1, L).unsqueeze(1)  # [N,1,L]


def _safe_torch_load(path, map_location="cpu", trust=False):
    try:
        # weights_only=True (PyTorch 2.6)
        return torch.load(path, map_location=map_location)
    except Exception as e1:
        try:
            with torch.serialization.safe_globals(_SAFE_NUMPY_GLOBALS):
                return torch.load(path, map_location=map_location)
        except Exception as e2:
            if trust:
                print(
                    f"[WARN] {path}: usando weights_only=False (trust_target_checkpoints=True)")
                return torch.load(path, map_location=map_location, weights_only=False)
            raise RuntimeError(
                f"No se pudo cargar de forma segura {path}. "
                f"Activa --trust si confías en el archivo.\n"
                f"Errores: (1) {e1}\n(2) {e2}"
            )

# =========================
#   Modelo (igual al de tu entrenamiento)
# =========================


def same_padding(kernel_size, dilation=1):
    return ((kernel_size - 1) * dilation) // 2


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, max(1, channels // r), kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(max(1, channels // r), channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.gate(s)
        return x * s


class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, dilation=1, dropout=0.0):
        super().__init__()
        pad = same_padding(k, dilation)
        self.depth = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad,
                               dilation=dilation, groups=in_ch, bias=False)
        self.point = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(
            32, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ResidualTCNBlock(nn.Module):
    def __init__(self, ch, k=7, dilations=(1, 2), dropout=0.05, use_se=True):
        super().__init__()
        layers, in_ch = [], ch
        for d in dilations:
            layers.append(DepthwiseSeparableConv1D(
                in_ch, ch, k=k, dilation=d, dropout=dropout))
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.se = SqueezeExcite1D(ch) if use_se else nn.Identity()

    def forward(self, x):
        out = self.net(x)
        out = self.se(out)
        return x + out


class BottleneckAttention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=ch, num_heads=num_heads, batch_first=False)
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        x_lbc = x.permute(2, 0, 1)  # [L,B,C]
        y, _ = self.attn(x_lbc, x_lbc, x_lbc, need_weights=False)
        y = self.ln(y)
        return x_lbc + y

    def to_conv(self, x_attn):
        return x_attn.permute(1, 2, 0)  # [B,C,L]


class ResUNetTCN(nn.Module):
    def __init__(self, in_ch=1, base=64, depth=3, k=7, dropout=0.05, heads=4):
        super().__init__()
        self.cfg = dict(in_ch=in_ch, base=base, depth=depth,
                        k=k, dropout=dropout, heads=heads)
        self.stem = nn.Conv1d(in_ch, base, kernel_size=3, padding=1)
        enc_blocks, downs = [], []
        ch = base
        for _ in range(depth):
            enc_blocks.append(ResidualTCNBlock(
                ch, k=k, dilations=(1, 2, 4), dropout=dropout))
            downs.append(nn.Conv1d(ch, ch*2, kernel_size=2, stride=2))
            ch *= 2
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downs = nn.ModuleList(downs)
        self.bottleneck = ResidualTCNBlock(
            ch, k=k, dilations=(1, 2, 4, 8), dropout=dropout)
        self.attn = BottleneckAttention(ch, num_heads=heads)
        dec_blocks, ups = [], []
        for _ in range(depth):
            ups.append(nn.ConvTranspose1d(ch, ch//2, kernel_size=2, stride=2))
            ch //= 2
            dec_blocks.append(ResidualTCNBlock(
                ch, k=k, dilations=(1, 2, 4), dropout=dropout))
        self.ups = nn.ModuleList(ups)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.proj = nn.Conv1d(base, 1, kernel_size=1)

    def _encode_features(self, x):
        skips = []
        h = self.stem(x)
        for blk, down in zip(self.enc_blocks, self.downs):
            h = blk(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        h = self.attn.to_conv(self.attn(h))   # [B,C,L']
        feat = torch.mean(h, dim=-1)          # [B,C]
        return h, feat, skips

    def forward(self, x):
        h, _, skips = self._encode_features(x)
        for up, blk in zip(self.ups, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                if diff > 0:
                    h = nn.functional.pad(h, (0, diff))
                else:
                    h = h[..., :skip.shape[-1]]
            h = h + skip
            h = blk(h)
        delta = self.proj(h)
        return x + delta

# =========================
#   Inferencia + plots
# =========================


def list_pt_files(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith((".pt", ".pth"))]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


@torch.no_grad()
def denoise_and_plot(folders, out_root, weights_path=None, max_total=100,
                     L_expected=512, device=None, trust=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    # Modelo
    model = ResUNetTCN(in_ch=1, base=64, depth=3, k=7,
                       dropout=0.05, heads=4).to(device)
    if weights_path and os.path.isfile(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[OK] Pesos cargados desde: {weights_path}")
    else:
        print("[WARN] No se cargaron pesos. El resultado será aleatorio (solo demo).")
    model.eval()

    # Salidas
    out_figs = os.path.join(out_root, "figs")
    out_data = os.path.join(out_root, "denoised")
    ensure_dir(out_figs)
    ensure_dir(out_data)

    processed = 0
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            print(f"[WARN] Carpeta inválida: {folder}")
            continue
        label = os.path.basename(os.path.abspath(folder))  # "eyem" o "musc"
        files = list_pt_files(folder)
        if not files:
            print(f"[WARN] Sin .pt en: {folder}")
            continue

        # carpetas específicas por dominio
        out_figs_dom = os.path.join(out_figs, label)
        out_data_dom = os.path.join(out_data, label)
        ensure_dir(out_figs_dom)
        ensure_dir(out_data_dom)

        for path in files:
            if processed >= max_total:
                break
            try:
                obj = _safe_torch_load(path, map_location="cpu", trust=trust)
                xN1L = _reshape_segments_to_N1L(
                    obj, L_expected=L_expected)  # [N,1,L]
            except Exception as e:
                print(f"[WARN] Saltando {path}: {e}")
                continue

            # Itera segmentos dentro del archivo, sin exceder el límite global
            for idx in range(xN1L.shape[0]):
                if processed >= max_total:
                    break
                x = xN1L[idx:idx+1]  # [1,1,L]
                x_dev = x.to(device)
                yhat = model(x_dev).cpu()  # [1,1,L]

                # ----- Guardar señal denoised -----
                base = os.path.splitext(os.path.basename(path))[0]
                out_name = f"{base}_seg{idx:03d}.pt"
                torch.save({"denoised": yhat.squeeze(0)},
                           os.path.join(out_data_dom, out_name))

                # ----- Plot lado a lado -----
                sig_in = x.squeeze().numpy()
                sig_out = yhat.squeeze().numpy()

                plt.figure(figsize=(10, 3.2), dpi=140)
                ax1 = plt.subplot(1, 2, 1)
                ax1.plot(sig_in)
                ax1.set_title(f"{label.upper()} - Entrada")
                ax1.set_xlabel("muestras")
                ax1.set_ylabel("amplitud")

                ax2 = plt.subplot(1, 2, 2)
                ax2.plot(sig_out)
                ax2.set_title("Inferencia (denoised)")
                ax2.set_xlabel("muestras")
                ax2.set_ylabel("amplitud")

                plt.tight_layout()
                fig_path = os.path.join(
                    out_figs_dom, f"{base}_seg{idx:03d}.png")
                plt.savefig(fig_path)
                plt.close()

                print(f"[OK] {label}: {base} seg{idx:03d} -> "
                      f"fig: {os.path.relpath(fig_path)}  |  out: {out_name}")
                processed += 1

        if processed >= max_total:
            break

    print(f"\n[DONE] Segmentos procesados: {processed}/{max_total}")
    print(f"Figuras en: {out_figs}")
    print(f"Denoised en: {out_data}")

# =========================
#   CLI
# =========================


def main():
    parser = argparse.ArgumentParser(
        description="Denoise y plots para ./eyem y ./musc (máx 100 segmentos en total).")
    parser.add_argument("--eyem", type=str, default="./eyem",
                        help="Carpeta con .pt de eyem")
    parser.add_argument("--musc", type=str, default="./musc",
                        help="Carpeta con .pt de musc")
    parser.add_argument("--weights", type=str, default="./best_joint_denoiser.pt",
                        help="Ruta a pesos del modelo (.pt con {'model': state_dict})")
    parser.add_argument("--max_total", type=int, default=100,
                        help="Máximo de segmentos a procesar (global)")
    parser.add_argument("--L", type=int, default=512,
                        help="Longitud esperada de ventana")
    parser.add_argument("--cpu", action="store_true", help="Forzar CPU")
    parser.add_argument("--trust", action="store_true",
                        help="Confiar en pickles (weights_only=False si falla)")
    parser.add_argument("--out", type=str, default="./outputs",
                        help="Carpeta raíz de salidas (figs y denoised)")
    args = parser.parse_args()

    dev = "cpu" if args.cpu else None
    denoise_and_plot(
        folders=[args.eyem, args.musc],
        out_root=args.out,
        weights_path=args.weights,
        max_total=args.max_total,
        L_expected=args.L,
        device=dev,
        trust=args.trust
    )


if __name__ == "__main__":
    main()
