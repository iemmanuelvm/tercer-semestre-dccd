# train_eog_denoiser_stream.py
# Denoiser EOG→EEG: ResUNet-TCN + SE + Attention + Streaming (512) con EEG limpia (GT)
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib  # <- NO importes pyplot globalmente; lo haremos dentro tras fijar backend
from collections import deque

# --------------------------
# Dataset: usa prepare_data
# --------------------------
from data_preparation_runner import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


def build_datasets(combin_num=11, train_per=0.9, noise_type='EOG'):
    """
    Carga datos con prepare_data y devuelve:
      - train_ds, test_ds (aplanado por SNR)
      - X_test_eog, y_test_eog (tensores para DEMO streaming por SNR)
    """
    X_train_eog, y_train_eog, X_test_eog, y_test_eog = prepare_data(
        combin_num=combin_num, train_per=train_per, noise_type=noise_type
    )
    # a tensores
    X_train_eog = torch.FloatTensor(X_train_eog)  # [N,1,512]
    y_train_eog = torch.FloatTensor(y_train_eog)  # [N,1,512]
    X_test_eog = torch.FloatTensor(X_test_eog)    # [11,M,1,512]
    y_test_eog = torch.FloatTensor(y_test_eog)    # [11,M,1,512]

    print("X_train_eog:", X_train_eog.shape)
    print("y_train_eog:", y_train_eog.shape)
    print("X_test_eog: ", X_test_eog.shape)
    print("y_test_eog: ", y_test_eog.shape)

    # aplanado test para métrica agregada
    snr_levels, M, C, L = X_test_eog.shape
    X_test_flat = X_test_eog.reshape(snr_levels * M, C, L)
    y_test_flat = y_test_eog.reshape(snr_levels * M, C, L)

    train_ds = TensorDataset(X_train_eog, y_train_eog)
    test_ds = TensorDataset(X_test_flat, y_test_flat)
    return train_ds, test_ds, X_test_eog, y_test_eog


# --------------------------
# Modelo: ResUNet-TCN + SE + Atención
# --------------------------
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
        return x * self.gate(s)


class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, dilation=1, dropout=0.0):
        super().__init__()
        pad = same_padding(k, dilation)
        self.depth = nn.Conv1d(in_ch, in_ch, k, padding=pad, dilation=dilation,
                               groups=in_ch, bias=False)
        self.point = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(
            32, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.norm(x)
        x = self.act(x)
        return self.drop(x)


class ResidualTCNBlock(nn.Module):
    def __init__(self, ch, k=7, dilations=(1, 2), dropout=0.05, use_se=True):
        super().__init__()
        layers, in_ch = [], ch
        for d in dilations:
            layers += [DepthwiseSeparableConv1D(in_ch,
                                                ch, k=k, dilation=d, dropout=dropout)]
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.se = SqueezeExcite1D(ch) if use_se else nn.Identity()

    def forward(self, x):
        out = self.net(x)
        out = self.se(out)
        return x + out


class BottleneckAttention(nn.Module):
    """Autoatención ligera en el cuello para dependencias largas."""

    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=ch, num_heads=num_heads, batch_first=False)
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        # x: [B,C,L] -> attn espera [L,B,C]
        x_perm = x.permute(2, 0, 1)
        x2, _ = self.attn(x_perm, x_perm, x_perm, need_weights=False)
        x2 = self.ln(x2)
        return x_perm + x2

    def to_conv(self, x_attn):
        return x_attn.permute(1, 2, 0)


class ResUNetTCN(nn.Module):
    def __init__(self, in_ch=1, base=64, depth=3, k=7, dropout=0.05, heads=4):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, base, kernel_size=3, padding=1)
        # Encoder
        enc_blocks, downs, ch = [], [], base
        for _ in range(depth):
            enc_blocks.append(ResidualTCNBlock(
                ch, k=k, dilations=(1, 2, 4), dropout=dropout))
            downs.append(nn.Conv1d(ch, ch * 2, kernel_size=2, stride=2))
            ch *= 2
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downs = nn.ModuleList(downs)
        # Bottleneck
        self.bottleneck = ResidualTCNBlock(
            ch, k=k, dilations=(1, 2, 4, 8), dropout=dropout)
        self.attn = BottleneckAttention(ch, num_heads=heads)
        # Decoder
        dec_blocks, ups = [], []
        for _ in range(depth):
            ups.append(nn.ConvTranspose1d(
                ch, ch // 2, kernel_size=2, stride=2))
            ch //= 2
            dec_blocks.append(ResidualTCNBlock(
                ch, k=k, dilations=(1, 2, 4), dropout=dropout))
        self.ups = nn.ModuleList(ups)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.proj = nn.Conv1d(base, 1, kernel_size=1)

    def forward(self, x):
        # x: [B,1,L]
        skips, h = [], self.stem(x)
        for blk, down in zip(self.enc_blocks, self.downs):
            h = blk(h)
            skips.append(h)
            h = down(h)  # /2
        h = self.bottleneck(h)
        h = self.attn.to_conv(self.attn(h))
        for up, blk in zip(self.ups, self.dec_blocks):
            h = up(h)  # *2
            skip = skips.pop()
            # seguridad de longitud
            if h.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                if diff > 0:
                    h = nn.functional.pad(h, (0, diff))
                else:
                    h = h[..., :skip.shape[-1]]
            h = h + skip
            h = blk(h)
        delta = self.proj(h)
        y_hat = x + delta  # residual
        return y_hat


# --------------------------
# Métricas + evaluación
# --------------------------
@torch.no_grad()
def compute_metrics(y_true, y_pred, eps=1e-8):
    diff = y_pred - y_true
    mse = torch.mean(diff ** 2)
    rmse = torch.sqrt(mse + eps)
    rms_true = torch.sqrt(torch.mean(y_true ** 2) + eps)
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
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        preds.append(yhat.detach().cpu())
        gts.append(yb.detach().cpu())
    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(gts, dim=0)
    return compute_metrics(y_true, y_pred)


# --------------------------
# Entrenamiento
# --------------------------
def train_model(
    train_ds,
    test_ds,
    epochs=50,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    model_save_path="./best_eog_denoiser.pt",
):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = ResUNetTCN(in_ch=1, base=64, depth=3, k=7,
                       dropout=0.05, heads=4).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    best_val = math.inf
    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            loss = criterion(yhat, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:03d} "
            f"| Train -> MSE: {train_metrics['MSE']:.6f}, RMSE: {train_metrics['RMSE']:.6f}, "
            f"RRMSE: {train_metrics['RRMSE']:.6f}, CC: {train_metrics['CC']:.4f} "
            f"| Test -> MSE: {test_metrics['MSE']:.6f}, RMSE: {test_metrics['RMSE']:.6f}, "
            f"RRMSE: {test_metrics['RRMSE']:.6f}, CC: {test_metrics['CC']:.4f}"
        )

        if test_metrics["MSE"] < best_val:
            best_val = test_metrics["MSE"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {"base": 64, "depth": 3, "k": 7, "dropout": 0.05, "heads": 4},
                },
                model_save_path,
            )

    print(f"\nTraining done. Best test MSE: {best_val:.6f}")
    return model


# --------------------------
# Carga modelo entrenado
# --------------------------
@torch.no_grad()
def load_trained_model(model_path="./best_eog_denoiser.pt", device=device):
    model = ResUNetTCN(in_ch=1, base=64, depth=3, k=7,
                       dropout=0.05, heads=4).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# --------------------------
# Generadores de stream (512)
# --------------------------
def stream_from_test_windows_pair(X_test_level, Y_test_level):
    """
    Stream emparejado de prueba (por SNR): yield (noisy_512, clean_512)
    X_test_level, Y_test_level: tensores [M,1,512]
    """
    X_np = X_test_level.detach().cpu().numpy()
    Y_np = Y_test_level.detach().cpu().numpy()
    M = min(X_np.shape[0], Y_np.shape[0])
    for i in range(M):
        yield (X_np[i, 0, :], Y_np[i, 0, :])


def stream_from_numpy_1d(signal_1d, chunk_size=512):
    """Para un vector 1D largo -> bloques consecutivos de chunk_size."""
    N = len(signal_1d)
    for start in range(0, N, chunk_size):
        end = start + chunk_size
        if end > N:
            last = np.zeros(chunk_size, dtype=np.float32)
            tail = np.asarray(signal_1d[start:], dtype=np.float32)
            last[: tail.shape[0]] = tail
            yield last
            break
        yield np.asarray(signal_1d[start:end], dtype=np.float32)


def stream_from_numpy_pair(noisy_1d, clean_1d, chunk_size=512):
    """
    Para dos vectores 1D (noisy, clean) -> bloques emparejados (noisy_512, clean_512)
    """
    N = min(len(noisy_1d), len(clean_1d))
    for start in range(0, N, chunk_size):
        end = start + chunk_size
        n_blk = np.zeros(chunk_size, dtype=np.float32)
        c_blk = np.zeros(chunk_size, dtype=np.float32)
        n_tail = np.asarray(noisy_1d[start:min(end, N)], dtype=np.float32)
        c_tail = np.asarray(clean_1d[start:min(end, N)], dtype=np.float32)
        n_blk[: n_tail.shape[0]] = n_tail
        c_blk[: c_tail.shape[0]] = c_tail
        yield (n_blk, c_blk)
        if end >= N:
            break


# --------------------------
# Plot en tiempo real (ruidosa, denoised, limpia)
# --------------------------
ANIM_HOLD = None  # mantiene viva la animación para evitar garbage collection


def _ensure_interactive_backend():
    """
    Si el backend no es interactivo (Agg/Inline), intentamos cambiar a QtAgg/TkAgg.
    Debe llamarse ANTES de importar matplotlib.pyplot o crear figuras.
    """
    backend = matplotlib.get_backend().lower()
    if "agg" in backend or "inline" in backend:
        for cand in ("QtAgg", "TkAgg"):
            try:
                matplotlib.use(cand, force=True)
                print(
                    f"[plot] Backend no interactivo detectado ({backend}). Cambiado a {cand}.")
                return
            except Exception:
                continue
        print(
            f"[plot] Backend no interactivo: {backend}. No se pudo cambiar automáticamente.")
        print(
            "       Sugerencia: instala PyQt5 o Tkinter, o ejecuta en un entorno con GUI.")
    else:
        print(f"[plot] Backend: {matplotlib.get_backend()}")


def realtime_infer_plot(
    model,
    sample_stream,
    chunk_size=512,
    fs=256,          # ajusta a tu muestreo real
    history_sec=10,
    device=device,
):
    """
    sample_stream debe yield:
      - (noisy_512,)  -> grafica ruidosa + denoised
      - (noisy_512, clean_512) -> añade curva de EEG limpia (GT)
    Devuelve (fig, anim). NO llama plt.show(); lo hacemos fuera.
    """
    _ensure_interactive_backend()
    # Importamos pyplot y animación después de fijar backend
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    history_len = int(history_sec * fs)
    buf_raw, buf_den, buf_gt = deque(maxlen=history_len), deque(
        maxlen=history_len), deque(maxlen=history_len)
    has_gt = False

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    line_raw, = ax.plot([], [], lw=1.0, label="Noisy (EEG+EOG)")
    line_den, = ax.plot([], [], lw=1.2, label="Denoised EEG")
    line_gt,  = ax.plot([], [], lw=1.2, linestyle="--",
                        label="EEG limpia (GT)")
    line_gt.set_visible(False)

    ax.set_title(
        f"Streaming denoising – actualización cada {chunk_size} muestras")
    ax.set_xlabel("Muestra")
    ax.set_ylabel("Amplitud")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    def init():
        line_raw.set_data([], [])
        line_den.set_data([], [])
        line_gt.set_data([], [])
        ax.set_xlim(0, max(1, history_len))
        ax.set_ylim(-1, 1)
        return line_raw, line_den, line_gt

    @torch.no_grad()
    def update(_i):
        nonlocal has_gt
        try:
            item = next(sample_stream)
        except StopIteration:
            return line_raw, line_den, line_gt

        if isinstance(item, (tuple, list)) and len(item) >= 1:
            noisy = np.asarray(item[0])
            gt = np.asarray(item[1]) if len(item) > 1 else None
        else:
            noisy = np.asarray(item)
            gt = None

        assert noisy.shape[0] == chunk_size, f"Bloque esperado de {chunk_size}."
        if gt is not None:
            assert gt.shape[0] == chunk_size, "GT debe tener el mismo tamaño de bloque."

        xb = torch.from_numpy(
            noisy[None, None, :].astype(np.float32)).to(device)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            yhat = model(xb).detach().cpu().numpy()[0, 0]

        buf_raw.extend(noisy.tolist())
        buf_den.extend(yhat.tolist())
        if gt is not None:
            buf_gt.extend(gt.tolist())
            has_gt = True

        raw = np.asarray(buf_raw, dtype=np.float32)
        den = np.asarray(buf_den, dtype=np.float32)
        x0 = max(0, len(raw) - history_len)
        x = np.arange(x0, x0 + min(history_len, len(raw)))
        raw_win = raw[-history_len:]
        den_win = den[-history_len:]

        line_raw.set_data(x, raw_win)
        line_den.set_data(x, den_win)

        if has_gt and len(buf_gt) > 0:
            gt_arr = np.asarray(buf_gt, dtype=np.float32)[-history_len:]
            line_gt.set_data(x, gt_arr)
            if not line_gt.get_visible():
                line_gt.set_visible(True)
            vis = np.concatenate([raw_win, den_win, gt_arr])
        else:
            vis = np.concatenate([raw_win, den_win])

        ax.set_xlim(max(0, len(raw) - history_len), max(history_len, len(raw)))
        m = float(np.max(np.abs(vis))) + 1e-6
        ax.set_ylim(-1.2 * m, 1.2 * m)
        return line_raw, line_den, line_gt

    interval_ms = int(1000 * chunk_size / max(1, fs))
    anim = FuncAnimation(fig, update, init_func=init,
                         interval=interval_ms, blit=False)
    return fig, anim


def run_realtime_demo_from_test(
    model_path="./best_eog_denoiser.pt",
    snr_idx=5,
    fs=256,
    history_sec=10,
    X_test_eog=None,
    y_test_eog=None,
):
    """
    Simula stream con (noisy, clean) usando el set de prueba (por SNR).
    """
    assert X_test_eog is not None and y_test_eog is not None, \
        "Debes construir el dataset con prepare_data primero."
    model = load_trained_model(model_path, device=device)

    # Generador emparejado
    sample_stream = stream_from_test_windows_pair(
        X_test_eog[snr_idx], y_test_eog[snr_idx])

    fig, anim = realtime_infer_plot(
        model,
        sample_stream,
        chunk_size=512,
        fs=fs,
        history_sec=history_sec,
        device=device,
    )

    # Mantén una referencia global para evitar garbage collection
    global ANIM_HOLD
    ANIM_HOLD = anim

    # Importamos pyplot aquí (ya con backend fijado)
    import matplotlib.pyplot as plt
    fig.tight_layout()
    plt.show()
    return anim


# --------------------------
# Ejecución principal
# --------------------------
if __name__ == "__main__":
    # 1) Construye SIEMPRE el dataset con prepare_data:
    train_ds, test_ds, X_test_eog, y_test_eog = build_datasets(
        combin_num=11, train_per=0.9, noise_type="EMG"
    )

    # 2) (opcional) Entrenar:
    # _ = train_model(
    #     train_ds, test_ds,
    #     epochs=300, batch_size=256, lr=1e-3, weight_decay=1e-4,
    #     model_save_path="./best_eog_denoiser.pt"
    # )

    # 3) Demo de inferencia + animación por bloques de 512 con EEG limpia (GT):
    run_realtime_demo_from_test(
        model_path="./best_emg_denoiser.pt",
        snr_idx=5,     # elige nivel SNR [0..10]
        fs=512,        # ajusta a tu frecuencia real
        history_sec=1,
        X_test_eog=X_test_eog,
        y_test_eog=y_test_eog,
    )
