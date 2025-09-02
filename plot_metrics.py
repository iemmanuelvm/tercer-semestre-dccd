import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


def plot_regression_metrics(
    csv_path: str,
    language: str = "es",
    save_dir: str | None = None,
):
    plt.rc("font", family="Times New Roman")

    df = pd.read_csv(csv_path)

    # Limpiar encabezados y posibles espacios en números
    df.columns = df.columns.str.strip()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")

    num_cols = [
        "train_mse", "train_rmse", "train_rrmse", "train_cc",
        "test_mse", "test_rmse", "test_rrmse", "test_cc", "ot_lambda_alpha"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    texts = {
        "es": {
            "title": "Entrenamiento vs Prueba",
            "x": "Épocas",
            "y": {
                "mse": "Error cuadrático medio (MSE)",
                "rmse": "Raíz del ECM (RMSE)",
                "rrmse": "RRMSE",
                "cc": "Correlación (CC)",
                "lambda": "λ (OT)",
            },
            "legend": {"train": "Entrenamiento", "test": "Prueba", "lambda": "λ (OT)"},
        },
        "en": {
            "title": "Training vs Test",
            "x": "Epochs",
            "y": {
                "mse": "Mean Squared Error (MSE)",
                "rmse": "Root MSE (RMSE)",
                "rrmse": "Relative RMSE (RRMSE)",
                "cc": "Correlation (CC)",
                "lambda": "λ (OT)",
            },
            "legend": {"train": "Train", "test": "Test", "lambda": "λ (OT)"},
        },
    }
    t = texts["es" if language == "es" else "en"]

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    line_kw_train = dict(linewidth=1, linestyle="-", marker="x", markersize=5)
    line_kw_test = dict(linewidth=1, linestyle="--", marker="s", markersize=5)
    line_kw_lambda = dict(linewidth=1, linestyle=":", marker="d", markersize=5)

    x = df["epoch"]

    # MSE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, df["train_mse"], label=t["legend"]["train"], **line_kw_train)
    ax.plot(x, df["test_mse"],  label=t["legend"]["test"],  **line_kw_test)
    ax.set_title(f'{t["title"]} — MSE\n', fontsize=18, fontweight="bold")
    ax.set_xlabel(t["x"], fontsize=18, fontweight="bold")
    ax.set_ylabel(t["y"]["mse"], fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14, frameon=True, fancybox=True)
    ax.grid(True)
    if save_dir:
        fig.savefig(os.path.join(save_dir, "metric_mse.png"),
                    bbox_inches="tight", dpi=300)

    # RMSE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, df["train_rmse"], label=t["legend"]["train"], **line_kw_train)
    ax.plot(x, df["test_rmse"],  label=t["legend"]["test"],  **line_kw_test)
    ax.set_title(f'{t["title"]} — RMSE\n', fontsize=18, fontweight="bold")
    ax.set_xlabel(t["x"], fontsize=18, fontweight="bold")
    ax.set_ylabel(t["y"]["rmse"], fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14, frameon=True, fancybox=True)
    ax.grid(True)
    if save_dir:
        fig.savefig(os.path.join(save_dir, "metric_rmse.png"),
                    bbox_inches="tight", dpi=300)

    # RRMSE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, df["train_rrmse"], label=t["legend"]["train"], **line_kw_train)
    ax.plot(x, df["test_rrmse"],  label=t["legend"]["test"],  **line_kw_test)
    ax.set_title(f'{t["title"]} — RRMSE\n', fontsize=18, fontweight="bold")
    ax.set_xlabel(t["x"], fontsize=18, fontweight="bold")
    ax.set_ylabel(t["y"]["rrmse"], fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14, frameon=True, fancybox=True)
    ax.grid(True)
    if save_dir:
        fig.savefig(os.path.join(save_dir, "metric_rrmse.png"),
                    bbox_inches="tight", dpi=300)

    # CC (solo CC, sin twinx)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, df["train_cc"], label=t["legend"]["train"], **line_kw_train)
    ax.plot(x, df["test_cc"],  label=t["legend"]["test"],  **line_kw_test)
    ax.set_title(f'{t["title"]} — CC\n', fontsize=18, fontweight="bold")
    ax.set_xlabel(t["x"], fontsize=18, fontweight="bold")
    ax.set_ylabel(t["y"]["cc"], fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14, frameon=True, fancybox=True)
    ax.grid(True)
    if save_dir:
        fig.savefig(os.path.join(save_dir, "metric_cc.png"),
                    bbox_inches="tight", dpi=300)

    # λ (OT) en gráfica aparte
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, df["ot_lambda_alpha"], label=t["legend"]
            ["lambda"], **line_kw_lambda)
    ax.set_title("λ (OT)\n", fontsize=18, fontweight="bold")
    ax.set_xlabel(t["x"], fontsize=18, fontweight="bold")
    ax.set_ylabel(t["y"]["lambda"], fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14, frameon=True, fancybox=True)
    ax.grid(True)
    if save_dir:
        fig.savefig(os.path.join(save_dir, "metric_lambda.png"),
                    bbox_inches="tight", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_regression_metrics(
        "training_metrics.csv",
        language="es",
        save_dir="plots",
    )
