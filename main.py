import numpy as np
import torch
from torch.utils.data import TensorDataset

from data_preparation_runner import prepare_data
from utils.tensors import to_tensor, flatten_snr
from utils.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

X_train_EMG, y_train_EMG, X_test_EMG, y_test_EMG = prepare_data(
    combin_num=11, train_per=0.9, noise_type="EMG"
)
X_train_EOG, y_train_EOG, X_test_EOG, y_test_EOG = prepare_data(
    combin_num=11, train_per=0.9, noise_type="EOG"
)

X_train_EMG = to_tensor(X_train_EMG)
y_train_EMG = to_tensor(y_train_EMG)
X_test_EMG = to_tensor(X_test_EMG)
y_test_EMG = to_tensor(y_test_EMG)

X_train_EOG = to_tensor(X_train_EOG)
y_train_EOG = to_tensor(y_train_EOG)
X_test_EOG = to_tensor(X_test_EOG)
y_test_EOG = to_tensor(y_test_EOG)

assert X_train_EMG.ndim == 3 and y_train_EMG.ndim == 3, "Train EMG must be [N,1,L]"
assert X_train_EOG.ndim == 3 and y_train_EOG.ndim == 3, "Train EOG must be [N,1,L]"
assert X_test_EMG.ndim == 4 and y_test_EMG.ndim == 4, "Test EMG must be [SNR,M,1,L]"
assert X_test_EOG.ndim == 4 and y_test_EOG.ndim == 4, "Test EOG must be [SNR,M,1,L]"
assert X_train_EMG.shape[1] == 1 and X_train_EOG.shape[1] == 1, "Channels must be 1"

L = X_train_EMG.shape[-1]
assert L == X_train_EOG.shape[-1] == y_train_EMG.shape[-1] == y_train_EOG.shape[-1], "Lengths must match"

print(f"[INFO] Window length L={L}")
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

X_test_EMG_flat, y_test_EMG_flat = flatten_snr(X_test_EMG, y_test_EMG)
X_test_EOG_flat, y_test_EOG_flat = flatten_snr(X_test_EOG, y_test_EOG)

X_test_joint = torch.cat([X_test_EMG_flat, X_test_EOG_flat], dim=0)
y_test_joint = torch.cat([y_test_EMG_flat, y_test_EOG_flat], dim=0)

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
        target_dirs=("./eyem", "./musc"),
        ot_kind="sinkhorn",
        ot_weight=0.1,
        ramp_epochs=20,
        ot_blur=0.05
    )
