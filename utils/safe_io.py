import os
import numpy as np
import torch


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


def _reshape_segments_to_N1L(obj, L_expected: int) -> torch.Tensor:
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
        ) == L_expected, f"Segment length {t.numel()} != {L_expected}"
        return t.view(1, 1, L_expected)
    if t.shape[-1] == L_expected:
        return t.reshape(-1, L_expected).unsqueeze(1)
    if t.shape[0] == L_expected and t.ndim >= 2:
        t = t.permute(*range(1, t.ndim), 0).contiguous()
        return t.reshape(-1, L_expected).unsqueeze(1)
    raise ValueError(
        f"Could not infer [*,*,{L_expected}] from shape {tuple(t.shape)}")


def _safe_torch_load(path, map_location="cpu", trust: bool = False):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e1:
        try:
            with torch.serialization.safe_globals(_SAFE_NUMPY_GLOBALS):
                return torch.load(path, map_location=map_location)
        except Exception as e2:
            if trust:
                print(
                    f"[WARN] {path}: using weights_only=False (trust_target_checkpoints=True)")
                return torch.load(path, map_location=map_location, weights_only=False)
            raise RuntimeError(
                f"Could not safely load {path}. "
                f"Enable trust_target_checkpoints=True if you trust the file.\n"
                f"Errors: (1) {e1}\n(2) {e2}"
            )


def load_target_pt_folders(folders, L_expected=512, trust_target_checkpoints=False):
    paths = []
    for d in folders:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".pt", ".pth")):
                    paths.append(os.path.join(d, f))
    if not paths:
        print("[WARN] No .pt files found in", folders)
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
            print(f"[WARN] Skipping {p}: {e}")

    if not segments:
        print("[WARN] Could not load any valid .pt files.")
        return torch.empty(0, 1, L_expected)

    X_tgt = torch.cat(segments, dim=0)
    print(f"[INFO] Target loaded: {X_tgt.shape[0]} segments from {len(paths)} files "
          f"({bad} failed). Shape={tuple(X_tgt.shape)}")
    return X_tgt
