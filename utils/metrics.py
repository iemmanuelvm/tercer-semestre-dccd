import torch
from torch.utils.data import TensorDataset, DataLoader


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
