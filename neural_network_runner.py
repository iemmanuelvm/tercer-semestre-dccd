import math
import torch
import torch.nn as nn
from typing import Optional

# ---------------------------------------------------------
# Positional Encoding (senoidal) para tokens 1D


class SinePositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        """
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


# ---------------------------------------------------------
# Transformer 1D con patching
class Transformer1D(nn.Module):
    """
    Red Transformer 1D totalmente independiente.

    I/O por defecto:
      - Entrada:  [B, in_ch, L]  (p.ej., [B, 1, 512])
      - Salida:   [B, out_ch, L] (p.ej., [B, 1, 512])

    Detalles:
      - Patch embedding 1D: Conv1d(in_ch -> d_model, stride=patch_size)
      - TransformerEncoder (batch_first=True)
      - Unpatch: ConvTranspose1d(d_model -> out_ch, stride=patch_size)
      - Recorta/pad para conservar exactamente la longitud original.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        d_model: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        patch_size: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        causal: bool = False,          # True = máscara causal (autoregresivo)
        use_residual_in_out: bool = True,  # skip global si in_ch==out_ch
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.patch_size = patch_size
        self.causal = causal
        self.use_residual_in_out = use_residual_in_out and (in_ch == out_ch)

        # Embedding de patches (tokens)
        # Conv1d con kernel=stride=patch_size para dividir la secuencia
        self.patch = nn.Conv1d(
            in_ch, d_model, kernel_size=patch_size, stride=patch_size)

        # Positional encoding
        self.posenc = SinePositionalEncoding1D(d_model)

        # Capa base del encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,   # trabaja con [B, L, C]
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Proyección de salida + unpatch (de tokens a señal)
        self.unpatch = nn.ConvTranspose1d(
            d_model, out_ch, kernel_size=patch_size, stride=patch_size
        )

        # Normalización final opcional (suele estabilizar)
        self.out_norm = nn.Identity() if out_ch == 1 else nn.BatchNorm1d(out_ch)

    def _build_causal_mask(self, L: int, device) -> torch.Tensor:
        # Máscara booleana Upper-Triangular para impedir mirar al futuro
        # shape: [L, L], True = posición enmascarada
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        """
        x:  [B, in_ch, L]
        src_key_padding_mask: [B, Lp] con True en posiciones a ignorar (opcional)
        """
        B, _, L = x.shape
        residual = x if self.use_residual_in_out else None

        # ---- Patchificar ----
        # tokens: [B, d_model, Lp]
        tokens = self.patch(x)
        Lp = tokens.size(-1)

        # ---- Preparar para Transformer ----
        tokens = tokens.transpose(1, 2)    # [B, Lp, d_model]
        tokens = self.posenc(tokens)       # + PE

        # Máscara causal si aplica
        src_mask = self._build_causal_mask(
            Lp, tokens.device) if self.causal else None

        # ---- Encoder ----
        # src_key_padding_mask: True = ignorar (por si paddeaste manualmente)
        tokens = self.encoder(
            # [B, Lp, d_model]
            tokens, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # ---- Volver a señal (unpatch) ----
        tokens = tokens.transpose(1, 2)    # [B, d_model, Lp]
        y = self.unpatch(tokens)           # [B, out_ch, L'] con L' ≈ L

        # Cortar o padear para igualar longitud exacta
        if y.size(-1) > L:
            y = y[..., :L]
        elif y.size(-1) < L:
            diff = L - y.size(-1)
            y = nn.functional.pad(y, (0, diff))

        # Skip global (opcional) para facilitar el entrenamiento
        if residual is not None:
            y = y + residual

        return self.out_norm(y)
