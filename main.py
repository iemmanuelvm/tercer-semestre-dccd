import numpy as np
import torch
from torch.utils.data import TensorDataset

from data_preparation_runner import prepare_data
from utils.tensors import to_tensor, flatten_snr
from utils.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

X_train_EMG,  y_train_EMG,  X_test_EMG,  y_test_EMG = prepare_data(
    combin_num=11, train_per=0.9, noise_type="EMG"
)
X_train_EOG,  y_train_EOG,  X_test_EOG,  y_test_EOG = prepare_data(
    combin_num=11, train_per=0.9, noise_type="EOG"
)
X_train_CHEW, y_train_CHEW, X_test_CHEW, y_test_CHEW = prepare_data(
    combin_num=11, train_per=0.9, noise_type="CHEW"
)
X_train_SHIV, y_train_SHIV, X_test_SHIV, y_test_SHIV = prepare_data(
    combin_num=11, train_per=0.9, noise_type="SHIV"
)

X_train_EMG = to_tensor(X_train_EMG)
y_train_EMG = to_tensor(y_train_EMG)
X_test_EMG = to_tensor(X_test_EMG)
y_test_EMG = to_tensor(y_test_EMG)

X_train_EOG = to_tensor(X_train_EOG)
y_train_EOG = to_tensor(y_train_EOG)
X_test_EOG = to_tensor(X_test_EOG)
y_test_EOG = to_tensor(y_test_EOG)

X_train_CHEW = to_tensor(X_train_CHEW)
y_train_CHEW = to_tensor(y_train_CHEW)
X_test_CHEW = to_tensor(X_test_CHEW)
y_test_CHEW = to_tensor(y_test_CHEW)

X_train_SHIV = to_tensor(X_train_SHIV)
y_train_SHIV = to_tensor(y_train_SHIV)
X_test_SHIV = to_tensor(X_test_SHIV)
y_test_SHIV = to_tensor(y_test_SHIV)

for name, Xtr, ytr, Xte, yte in [
    ("EMG",  X_train_EMG,  y_train_EMG,  X_test_EMG,  y_test_EMG),
    ("EOG",  X_train_EOG,  y_train_EOG,  X_test_EOG,  y_test_EOG),
    ("CHEW", X_train_CHEW, y_train_CHEW, X_test_CHEW, y_test_CHEW),
    ("SHIV", X_train_SHIV, y_train_SHIV, X_test_SHIV, y_test_SHIV),
]:
    assert Xtr.ndim == 3 and ytr.ndim == 3, f"Train {name} must be [N,1,L]"
    assert Xte.ndim == 4 and yte.ndim == 4, f"Test {name} must be [SNR,M,1,L]"
    assert Xtr.shape[1] == 1,               f"Channels for {name} must be 1"

L = X_train_EMG.shape[-1]
for t in [
    X_train_EOG, y_train_EMG, y_train_EOG,
    X_train_CHEW, y_train_CHEW,
    X_train_SHIV, y_train_SHIV
]:
    assert t.shape[-1] == L, "Lengths must match across all artifacts"

print(f"[INFO] Window length L={L}")
def _sh(s): return tuple(s.shape)


print("X_train_EMG :", _sh(X_train_EMG))
print("y_train_EMG :", _sh(y_train_EMG))
print("X_test_EMG  :", _sh(X_test_EMG))
print("y_test_EMG  :", _sh(y_test_EMG))
print("X_train_EOG :", _sh(X_train_EOG))
print("y_train_EOG :", _sh(y_train_EOG))
print("X_test_EOG  :", _sh(X_test_EOG))
print("y_test_EOG  :", _sh(y_test_EOG))
print("X_train_CHEW:", _sh(X_train_CHEW))
print("y_train_CHEW:", _sh(y_train_CHEW))
print("X_test_CHEW :", _sh(X_test_CHEW))
print("y_test_CHEW :", _sh(y_test_CHEW))
print("X_train_SHIV:", _sh(X_train_SHIV))
print("y_train_SHIV:", _sh(y_train_SHIV))
print("X_test_SHIV :", _sh(X_test_SHIV))
print("y_test_SHIV :", _sh(y_test_SHIV))

X_train_joint = torch.cat(
    [X_train_EMG, X_train_EOG, X_train_CHEW, X_train_SHIV], dim=0
)
y_train_joint = torch.cat(
    [y_train_EMG, y_train_EOG, y_train_CHEW, y_train_SHIV], dim=0
)

X_test_EMG_flat,  y_test_EMG_flat = flatten_snr(X_test_EMG,  y_test_EMG)
X_test_EOG_flat,  y_test_EOG_flat = flatten_snr(X_test_EOG,  y_test_EOG)
X_test_CHEW_flat, y_test_CHEW_flat = flatten_snr(X_test_CHEW, y_test_CHEW)
X_test_SHIV_flat, y_test_SHIV_flat = flatten_snr(X_test_SHIV, y_test_SHIV)

X_test_joint = torch.cat(
    [X_test_EMG_flat, X_test_EOG_flat, X_test_CHEW_flat, X_test_SHIV_flat], dim=0
)
y_test_joint = torch.cat(
    [y_test_EMG_flat, y_test_EOG_flat, y_test_CHEW_flat, y_test_SHIV_flat], dim=0
)

train_ds = TensorDataset(X_train_joint, y_train_joint)
test_ds = TensorDataset(X_test_joint,  y_test_joint)

print("[INFO] Joint train:", tuple(X_train_joint.shape),
      " | Joint test:", tuple(X_test_joint.shape))

if __name__ == "__main__":
    _ = train_model(
        train_ds=train_ds,
        test_ds=test_ds,
        L=L,
        device=device,
        X_test_EMG=X_test_EMG,
        y_test_EMG=y_test_EMG,
        X_test_EOG=X_test_EOG,
        y_test_EOG=y_test_EOG,
        epochs=100,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        model_save_path="./best_joint_denoiser.pt",
        eval_per_snr=False,
        domain_adapt=True,
        target_dirs=("./eyem", "./musc", "./shiv", "./chew"),
        ot_kind="sinkhorn",
        ot_weight=0.1,
        ramp_epochs=20,
        ot_blur=0.05
    )
