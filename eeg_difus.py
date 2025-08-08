# eeg_difus.py
import math
from typing import List, Tuple
import torch
import torch.nn as nn

# -----------------------------
# Bloques básicos


def _norm1d(C: int):
    groups = 8 if C >= 8 else 1
    return nn.GroupNorm(num_groups=groups, num_channels=C)


class LocalInfoUnit(nn.Module):
    """Conv1d 1x3 -> hidden_dim (2 capas) + GELU."""

    def __init__(self, in_ch: int = 1, hidden_dim: int = 64, k: int = 3):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden_dim, kernel_size=k, padding=p),
            _norm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=p),
            _norm1d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):  # [B,1,L] -> [B,C,L]
        return self.net(x)


class FiLM(nn.Module):
    """Mapea sqrt(alpha_bar) -> (gamma, beta) por escala."""

    def __init__(self, n_scales: int, channels: int, hidden: int = 128):
        super().__init__()
        self.n_scales = n_scales
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * n_scales * channels)
        )

    def forward(self, sqrt_alpha_bar: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        sqrt_alpha_bar: [B]
        returns: [(gamma, beta)] * n_scales, cada uno [B, C, 1]
        """
        B = sqrt_alpha_bar.shape[0]
        gb = self.net(sqrt_alpha_bar.view(B, 1))  # [B, 2*S*C]
        gb = gb.view(B, self.n_scales, 2, self.channels)  # [B,S,2,C]
        outs = []
        for s in range(self.n_scales):
            gamma = gb[:, s, 0, :].unsqueeze(-1)  # [B,C,1]
            beta = gb[:, s, 1, :].unsqueeze(-1)
            outs.append((gamma, beta))
        return outs


class TransformerStack1D(nn.Module):
    """
    Pila de TransformerEncoderLayer con salida por capa (multi-escala por profundidad).
    Proyecta in_ch -> qkv_dim y de vuelta a in_ch.
    """

    def __init__(self, in_ch: int, qkv_dim: int = 32, heads: int = 1,
                 depth: int = 3, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        assert qkv_dim % heads == 0, "qkv_dim debe ser divisible por heads"
        self.proj_in = nn.Conv1d(in_ch, qkv_dim, kernel_size=1)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=qkv_dim, nhead=heads,
                dim_feedforward=ff_mult * qkv_dim,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(qkv_dim)
        self.proj_out = nn.Conv1d(qkv_dim, in_ch, kernel_size=1)
        self.depth = depth

    def forward(self, x):  # [B,C,L] -> (y:[B,C,L], feats:[list de B,C,L]*depth)
        B, C, L = x.shape
        h = self.proj_in(x).transpose(1, 2)  # [B,L,qkv]
        feats = []
        for lyr in self.layers:
            h = lyr(h)                        # [B,L,qkv]
            feats.append(self.proj_out(h.transpose(1, 2)))  # [B,C,L]
        h = self.norm(h)
        y = self.proj_out(h.transpose(1, 2))  # [B,C,L]
        return y, feats


class IntegrationModule(nn.Module):
    """Fusión multi-escala de las dos ramas + FiLM y proyección a 1 canal."""

    def __init__(self, in_ch: int = 64, k: int = 3, n_scales: int = 3):
        super().__init__()
        p = k // 2
        self.mix = nn.Sequential(
            nn.Conv1d(n_scales * in_ch, in_ch, kernel_size=1),
            _norm1d(in_ch),
            nn.GELU(),
            nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=p),
            _norm1d(in_ch),
            nn.GELU(),
        )
        self.to_out = nn.Conv1d(in_ch, 1, kernel_size=k, padding=p)

    def forward(self,
                scales_a: List[torch.Tensor],
                scales_b: List[torch.Tensor],
                film_params: List[Tuple[torch.Tensor, torch.Tensor]]):
        fused = []
        for (ha, hb), (gamma, beta) in zip(zip(scales_a, scales_b), film_params):
            h = ha + hb
            h = h * (1 + gamma) + beta   # broadcast en L
            fused.append(h)
        H = torch.cat(fused, dim=1)      # [B, S*C, L]
        H = self.mix(H)                  # [B, C, L]
        y = self.to_out(H)               # [B, 1, L]
        return y

# -----------------------------
# Modelo principal


class EEGDfusDenoiser(nn.Module):
    """
    ϵ_θ(x_t, x̃, √ᾱ) -> predicción de epsilon (mismo tamaño que x_t).
    Dos ramas: x_t y x̃; LIU -> TransformerStack -> fusión + FiLM.
    """

    def __init__(self,
                 in_ch: int = 1,
                 hidden_dim: int = 64,
                 heads: int = 1,
                 qkv_dim: int = 32,
                 depth: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.liu_a = LocalInfoUnit(in_ch, hidden_dim)
        self.liu_b = LocalInfoUnit(in_ch, hidden_dim)
        self.tr_a = TransformerStack1D(
            hidden_dim, qkv_dim, heads, depth, dropout=dropout)
        self.tr_b = TransformerStack1D(
            hidden_dim, qkv_dim, heads, depth, dropout=dropout)
        self.film = FiLM(n_scales=depth, channels=hidden_dim)
        self.fuse = IntegrationModule(in_ch=hidden_dim, n_scales=depth)

    def forward(self, x_t: torch.Tensor, x_tilde: torch.Tensor, sqrt_alpha_bar: torch.Tensor):
        # Rama x_t
        ha0 = self.liu_a(x_t)                  # [B,C,L]
        _, ha_scales = self.tr_a(ha0)          # lista [B,C,L]*depth
        # Rama x̃
        hb0 = self.liu_b(x_tilde)              # [B,C,L]
        _, hb_scales = self.tr_b(hb0)

        film_params = self.film(sqrt_alpha_bar)  # [(γ,β)]*depth, cada [B,C,1]
        y = self.fuse(ha_scales, hb_scales, film_params)  # [B,1,L]
        return y

# -----------------------------
# Difusión: schedule, training step y muestreador


def make_beta_schedule(T: int = 500, beta_start: float = 1e-4, beta_end: float = 2e-2):
    betas = torch.linspace(beta_start, beta_end, T)  # 1D [T]
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


@torch.no_grad()
def ddpm_sample(model: EEGDfusDenoiser,
                x_tilde: torch.Tensor,
                betas: torch.Tensor,
                alphas: torch.Tensor,
                alpha_bars: torch.Tensor,
                device=None):
    """
    Algoritmo 2 (DDPM) condicional en x̃.
    x_tilde: [B,1,L]
    """
    device = device or x_tilde.device
    B, _, L = x_tilde.shape
    y_t = torch.randn(B, 1, L, device=device)  # y_T ~ N(0,I)
    T = betas.shape[0]

    for t in range(T - 1, -1, -1):
        a_t = alphas[t]
        ab_t = alpha_bars[t]
        sqrt_ab_t_batch = torch.sqrt(ab_t).expand(B).to(device)
        eps = model(y_t, x_tilde, sqrt_ab_t_batch)  # [B,1,L]

        coef1 = 1.0 / torch.sqrt(a_t)
        coef2 = (1.0 - a_t) / torch.sqrt(1.0 - ab_t)
        y_tm1 = coef1 * (y_t - coef2 * eps)

        if t > 0:
            sigma_t = torch.sqrt(betas[t])
            y_tm1 = y_tm1 + sigma_t * torch.randn_like(y_tm1)

        y_t = y_tm1

    return y_t  # y_0


def diffusion_training_step(model: EEGDfusDenoiser,
                            x0: torch.Tensor,        # señal limpia [B,1,L]
                            # señal ruidosa (condición) [B,1,L]
                            x_tilde: torch.Tensor,
                            alpha_bars: torch.Tensor,
                            rng: torch.Generator = None):
    """
    Pérdida (13): ε ~ N(0,1), x_t = √ā x0 + √(1-ā) ε; MSE(ε̂, ε).
    """
    device = x0.device
    T = alpha_bars.shape[0]
    B = x0.size(0)

    # t uniforme, ᾱ continuo entre ᾱ_{t-1} y ᾱ_t
    t = torch.randint(low=1, high=T, size=(B,), device=device, generator=rng)
    ab_t = alpha_bars[t]
    ab_tm1 = alpha_bars[(t - 1).clamp(min=0)]
    u = torch.rand(B, device=device)
    a_bar = ab_tm1 * (1.0 - u) + ab_t * u         # [B]
    sqrt_a_bar = torch.sqrt(a_bar)
    sqrt_one_m_ab = torch.sqrt(1.0 - a_bar)

    eps = torch.randn_like(x0)
    x_t = sqrt_a_bar.view(B, 1, 1) * x0 + sqrt_one_m_ab.view(B, 1, 1) * eps

    eps_pred = model(x_t, x_tilde, sqrt_a_bar)    # [B,1,L]
    loss = nn.functional.mse_loss(eps_pred, eps)
    return loss
