import torch


def to_tensor(x):
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 2:
        t = t.unsqueeze(1)
    return t


def flatten_snr(X4, y4):
    SNR, M, C, L = X4.shape
    return X4.reshape(SNR * M, C, L), y4.reshape(SNR * M, C, L)
