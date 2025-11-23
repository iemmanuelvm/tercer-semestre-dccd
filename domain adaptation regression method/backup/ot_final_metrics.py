import math
import os
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_preparation_runner import prepare_data

# ---- GeomLoss ----
try:
    from geomloss import SamplesLoss
except ImportError as e:
    raise ImportError(
        "GeomLoss no está instalado. Instala con: pip install geomloss") from e

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# =========================
#   Datos fuente sintéticos
# =========================
# EMG
X_train_EMG, y_train_EMG, X_test_EMG, y_test_EMG = prepare_data(
    combin_num=11, train_per=0.9, noise_type="EMG"
)
# EOG
X_train_EOG, y_train_EOG, X_test_EOG, y_test_EOG = prepare_data(
    combin_num=11, train_per=0.9, noise_type="EOG"
)

# --------------------------
#   X_train_*: [N, 1, 512]
#   y_train_*: [N, 1, 512]
#   X_test_* : [SNR=11, M, 1, 512]
#   y_test_* : [SNR=11, M, 1, 512]
# --------------------------


def to_tensor(x):
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 2:
        t = t.unsqueeze(1)
    return t


X_train_EMG = to_tensor(X_train_EMG)
y_train_EMG = to_tensor(y_train_EMG)
X_test_EMG = to_tensor(X_test_EMG)
y_test_EMG = to_tensor(y_test_EMG)

X_train_EOG = to_tensor(X_train_EOG)
y_train_EOG = to_tensor(y_train_EOG)
X_test_EOG = to_tensor(X_test_EOG)
y_test_EOG = to_tensor(y_test_EOG)

assert X_train_EMG.ndim == 3 and y_train_EMG.ndim == 3, "Train EMG debe ser [N,1,L]"
assert X_train_EOG.ndim == 3 and y_train_EOG.ndim == 3, "Train EOG debe ser [N,1,L]"
assert X_test_EMG.ndim == 4 and y_test_EMG.ndim == 4, "Test EMG debe ser [SNR,M,1,L]"
assert X_test_EOG.ndim == 4 and y_test_EOG.ndim == 4, "Test EOG debe ser [SNR,M,1,L]"
assert X_train_EMG.shape[1] == 1 and X_train_EOG.shape[1] == 1, "Canales debe ser 1"
L = X_train_EMG.shape[-1]
assert L == X_train_EOG.shape[-1] == y_train_EMG.shape[-1] == y_train_EOG.shape[-1], "Longitudes deben coincidir"

print(f"[INFO] Longitud de ventana L={L}")
print("X_train_EMG:", tuple(X_train_EMG.shape))
print("y_train_EMG:", tuple(y_train_EMG.shape))
print("X_test_EMG :", tuple(X_test_EMG.shape))
print("y_test_EMG :", tuple(y_test_EMG.shape))
print("X_train_EOG:", tuple(X_train_EOG.shape))
print("y_train_EOG:", tuple(y_train_EOG.shape))
print("X_test_EOG :", tuple(X_test_EOG.shape))
print("y_test_EOG :", tuple(y_test_EOG.shape))

X_train_joint = torch.cat([X_train_EMG, X_train_EOG], dim=0)
y_train_joint = torch.cat([y_train_EMG, y_train_EOG], dim=0)


def flatten_snr(X4, y4):
    SNR, M, C, L = X4.shape
    return X4.reshape(SNR * M, C, L), y4.reshape(SNR * M, C, L)


X_test_EMG_flat, y_test_EMG_flat = flatten_snr(X_test_EMG, y_test_EMG)
X_test_EOG_flat, y_test_EOG_flat = flatten_snr(X_test_EOG, y_test_EOG)
X_test_joint = torch.cat([X_test_EMG_flat, X_test_EOG_flat], dim=0)
y_test_joint = torch.cat([y_test_EMG_flat, y_test_EOG_flat], dim=0)

train_ds = TensorDataset(X_train_joint, y_train_joint)
test_ds = TensorDataset(X_test_joint,  y_test_joint)

print("[INFO] Joint train:", tuple(X_train_joint.shape),
      " | Joint test:", tuple(X_test_joint.shape))

# ==============================
#   Carga dominio target (.pt) segura (PyTorch 2.6 + NumPy 2.x)
# ==============================


def _collect_numpy_safe_globals():
    base = [
        np.ndarray, np.dtype, np.generic, np.number, np.bool_,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
        np.complex64, np.complex128,
    ]
    # Pickles antiguos
    try:
        base.append(np.core.multiarray._reconstruct)
    except Exception:
        pass
    # NumPy 2.x: dtypes paramétricos
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


def _reshape_segments_to_N1L(obj, L_expected: int) -> torch.Tensor:
    """Devuelve tensor [N,1,L]. Acepta tensor, array, lista o dict con claves comunes."""
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
        assert t.numel(
        ) == L_expected, f"Segmento de longitud {t.numel()} != {L_expected}"
        return t.view(1, 1, L_expected)
    if t.shape[-1] == L_expected:              # [N, L]
        return t.reshape(-1, L_expected).unsqueeze(1)   # [N,1,L]
    if t.shape[0] == L_expected and t.ndim >= 2:  # [L, ...]
        t = t.permute(*range(1, t.ndim), 0).contiguous()
        return t.reshape(-1, L_expected).unsqueeze(1)
    raise ValueError(
        f"No pude inferir [*,*,{L_expected}] desde shape {tuple(t.shape)}")


def _safe_torch_load(path, map_location="cpu", trust: bool = False):
    """
    1) weights_only=True (por defecto en PyTorch 2.6)
    2) Reintento con allowlist ampliada (safe_globals)
    3) (opcional si trust=True) weights_only=False  << POTENCIALMENTE PELIGROSO
    """
    try:
        return torch.load(path, map_location=map_location)  # weights_only=True
    except Exception as e1:
        try:
            with torch.serialization.safe_globals(_SAFE_NUMPY_GLOBALS):
                # weights_only=True
                return torch.load(path, map_location=map_location)
        except Exception as e2:
            if trust:
                print(
                    f"[WARN] {path}: usando weights_only=False (trust_target_checkpoints=True)")
                return torch.load(path, map_location=map_location, weights_only=False)
            raise RuntimeError(
                f"No se pudo cargar de forma segura {path}. "
                f"Activa trust_target_checkpoints=True si confías en el archivo.\n"
                f"Errores: (1) {e1}\n(2) {e2}"
            )


def load_target_pt_folders(folders, L_expected=512, trust_target_checkpoints=False):
    paths = []
    for d in folders:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".pt", ".pth")):
                    paths.append(os.path.join(d, f))
    if not paths:
        print("[WARN] No se encontraron archivos .pt en", folders)
        return torch.empty(0, 1, L_expected)

    segments, bad = [], 0
    for p in paths:
        try:
            obj = _safe_torch_load(p, map_location="cpu",
                                   trust=trust_target_checkpoints)
            segN1L = _reshape_segments_to_N1L(obj, L_expected)
            segments.append(segN1L)
        except Exception as e:
            bad += 1
            print(f"[WARN] Saltando {p}: {e}")

    if not segments:
        print("[WARN] No se pudo cargar ningún .pt válido.")
        return torch.empty(0, 1, L_expected)

    X_tgt = torch.cat(segments, dim=0)
    print(f"[INFO] Target cargado: {X_tgt.shape[0]} segmentos desde {len(paths)} archivos "
          f"({bad} fallidos). Shape={tuple(X_tgt.shape)}")
    return X_tgt

# =========================
#   Modelo
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
        layers = []
        in_ch = ch
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
        x_lbc = x.permute(2, 0, 1)
        y, _ = self.attn(x_lbc, x_lbc, x_lbc, need_weights=False)
        y = self.ln(y)
        return x_lbc + y

    def to_conv(self, x_attn):  # [L,B,C] -> [B,C,L]
        return x_attn.permute(1, 2, 0)


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
        feat = torch.mean(h, dim=-1)          # GAP -> [B,C]
        return h, feat, skips

    def encode(self, x):
        _, feat, _ = self._encode_features(x)
        return feat

    def forward(self, x, return_features=False):
        h, feat, skips = self._encode_features(x)
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
        out = x + delta
        if return_features:
            return out, feat
        return out

# =========================
#   Métricas / Eval
# =========================


def kl_divergence(p, q):
    epsilon = 1e-6
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return torch.sum(p * torch.log(p / q), dim=-1)


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd


def jensen_shannon_distance(p, q):
    jsd = jensen_shannon_divergence(p, q)
    jsd = torch.clamp(jsd, min=0.0)
    return torch.sqrt(jsd)


@torch.no_grad()
def compute_metrics(y_true, y_pred, eps=1e-8):
    diff = y_pred - y_true
    mse = torch.mean(diff**2)
    rmse = torch.sqrt(mse + eps)
    rms_true = torch.sqrt(torch.mean(y_true**2) + eps)
    rrmse = rmse / (rms_true + eps)

    yt = y_true.squeeze(1)
    yp = y_pred.squeeze(1)
    yt_m = yt.mean(dim=-1, keepdim=True)
    yp_m = yp.mean(dim=-1, keepdim=True)
    cov = ((yt - yt_m) * (yp - yp_m)).mean(dim=-1)
    std_t = yt.std(dim=-1) + eps
    std_p = yp.std(dim=-1) + eps
    cc = torch.mean(cov / (std_t * std_p))

    return {"MSE": mse.item(), "RMSE": rmse.item(), "RRMSE": rrmse.item(), "CC": cc.item()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)
        preds.append(yhat.detach().cpu())
        gts.append(yb.detach().cpu())
    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(gts, dim=0)
    return compute_metrics(y_true, y_pred)


@torch.no_grad()
def evaluate_per_snr(model, X_test_4D, y_test_4D, device, batch_size=512):
    model.eval()
    SNR = X_test_4D.shape[0]
    out = []
    for i in range(SNR):
        ds = TensorDataset(X_test_4D[i], y_test_4D[i])
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        out.append(evaluate(model, dl, device))
    return out

# =========================
#   Entrenamiento + adaptación (GeomLoss) + KL/JSD
# =========================


def train_model(
    epochs=120,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    model_save_path="./best_joint_denoiser.pt",
    eval_per_snr=False,
    # ---- Adaptación de dominio ----
    domain_adapt=True,
    target_dirs=("./eyem", "./musc"),
    ot_kind="sinkhorn",           # "sinkhorn" o "energy"
    ot_weight=0.1,                # α
    ramp_epochs=20,               # λ(t) ramp-up
    ot_blur=0.05,                 # parámetro del Sinkhorn
    feature_l2norm=True,
    trust_target_checkpoints=False
):
    # Loaders fuente
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, drop_last=False,
        pin_memory=torch.cuda.is_available()
    )

    # Loader target (no labels)
    X_target = load_target_pt_folders(
        target_dirs, L_expected=L, trust_target_checkpoints=trust_target_checkpoints)
    if X_target.numel() == 0:
        domain_adapt = False
        print("[WARN] No hay datos target válidos. Entrenamiento sin adaptación.")
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

    # GeomLoss
    if ot_kind not in ("sinkhorn", "energy"):
        raise ValueError("ot_kind debe ser 'sinkhorn' o 'energy'")
    samples_loss = SamplesLoss(
        loss=ot_kind, p=2, blur=ot_blur, debias=True if ot_kind == "sinkhorn" else False)

    best_val = math.inf
    print("\n[INFO] Inicio de entrenamiento conjunto (EMG+EOG) con adaptación OT=",
          ot_kind if domain_adapt else "desactivada")

    if domain_adapt:
        tgt_iter = iter(target_loader)

    for epoch in range(1, epochs + 1):
        model.train()

        # Acumuladores para métricas de divergencia por época
        jsd_acc = 0.0
        jsdist_acc = 0.0
        kl_st_acc = 0.0
        kl_ts_acc = 0.0
        div_steps = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # ---- Fuente: pérdida supervisada + features
            yhat, f_src = model(xb, return_features=True)
            sup_loss = criterion(yhat, yb)

            # ---- Target: features y pérdida OT
            if domain_adapt:
                try:
                    xt = next(tgt_iter)[0]
                except StopIteration:
                    tgt_iter = iter(target_loader)
                    xt = next(tgt_iter)[0]
                xt = xt.to(device)
                with torch.set_grad_enabled(True):
                    f_tgt = model.encode(xt)   # solo encoder -> más ligero

                # Normalización y OT
                if feature_l2norm:
                    f_src = nn.functional.normalize(f_src, dim=1)
                    f_tgt = nn.functional.normalize(f_tgt, dim=1)

                ot_loss = samples_loss(f_src, f_tgt)

                # ---- MÉTRICAS entre distribuciones de features (promedio de batch) ----
                p = torch.softmax(f_src, dim=1)  # [Bs, D]
                q = torch.softmax(f_tgt, dim=1)  # [Bt, D]

                p_bar = p.mean(dim=0, keepdim=True)  # [1, D]
                q_bar = q.mean(dim=0, keepdim=True)  # [1, D]

                kl_st = kl_divergence(p_bar, q_bar).mean()
                kl_ts = kl_divergence(q_bar, p_bar).mean()
                jsd = jensen_shannon_divergence(p_bar, q_bar).mean()
                jsdist = jensen_shannon_distance(p_bar, q_bar).mean()

                jsd_acc += jsd.item()
                jsdist_acc += jsdist.item()
                kl_st_acc += kl_st.item()
                kl_ts_acc += kl_ts.item()
                div_steps += 1

                # Ramp-up de la adaptación
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

        # Logging por época
        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model,  test_loader,  device)

        # Promedios de divergencias en la época
        if div_steps > 0:
            jsd_epoch = jsd_acc / div_steps
            jsdist_epoch = jsdist_acc / div_steps
            klst_epoch = kl_st_acc / div_steps
            klts_epoch = kl_ts_acc / div_steps
        else:
            jsd_epoch = jsdist_epoch = klst_epoch = klts_epoch = float("nan")

        msg = (
            f"Epoch {epoch:03d} | "
            f"Train -> MSE {train_metrics['MSE']:.6f}, RMSE {train_metrics['RMSE']:.6f}, "
            f"RRMSE {train_metrics['RRMSE']:.6f}, CC {train_metrics['CC']:.4f} | "
            f"Test -> MSE {test_metrics['MSE']:.6f}, RMSE {test_metrics['RMSE']:.6f}, "
            f"RRMSE {test_metrics['RRMSE']:.6f}, CC {test_metrics['CC']:.4f}"
        )
        if domain_adapt:
            msg += (
                f" | OT λ*α={lam*ot_weight:.3f} "
                f"| KL(p||q)={klst_epoch:.4f}, KL(q||p)={klts_epoch:.4f}, "
                f"JSD={jsd_epoch:.4f}, JSdist={jsdist_epoch:.4f}"
            )
        print(msg)

        if test_metrics["MSE"] < best_val:
            best_val = test_metrics["MSE"]
            torch.save({"model": model.state_dict(),
                        "config": model.cfg}, model_save_path)

        if eval_per_snr and (epoch % 10 == 0 or epoch == epochs):
            print("\n[INFO] Per-SNR EMG:")
            per_snr_emg = evaluate_per_snr(
                model, X_test_EMG, y_test_EMG, device)
            for i, m in enumerate(per_snr_emg):
                print(
                    f"  EMG SNR[{i:02d}] -> MSE {m['MSE']:.6f}, RMSE {m['RMSE']:.6f}, RRMSE {m['RRMSE']:.6f}, CC {m['CC']:.4f}")
            print("\n[INFO] Per-SNR EOG:")
            per_snr_eog = evaluate_per_snr(
                model, X_test_EOG, y_test_EOG, device)
            for i, m in enumerate(per_snr_eog):
                print(
                    f"  EOG SNR[{i:02d}] -> MSE {m['MSE']:.6f}, RMSE {m['RMSE']:.6f}, RRMSE {m['RRMSE']:.6f}, CC {m['CC']:.4f}")
            print("")

    print(f"\n[OK] Entrenamiento finalizado. Best test MSE: {best_val:.6f}")
    return model


if __name__ == "__main__":
    _ = train_model(
        epochs=100,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        model_save_path="./best_joint_denoiser.pt",
        eval_per_snr=False,
        # ---- knobs de adaptación ----
        domain_adapt=True,
        target_dirs=("./eyem", "./musc"),
        ot_kind="sinkhorn",      # también "energy" (MMD-like)
        ot_weight=0.1,
        ramp_epochs=20,
        ot_blur=0.05,
        feature_l2norm=True,
        trust_target_checkpoints=False
    )
