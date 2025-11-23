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

    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
        .str.lower()
    )

    if "epoch" not in df.columns:
        raise KeyError(
            "Falta la columna 'epoch' (después de normalización). "
            f"Columnas encontradas: {list(df.columns)}"
        )
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

    def plot_pair(y_train: str, y_test: str, title_suffix: str, y_label: str, fname: str):
        if y_train in df.columns and y_test in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, df[y_train], label=t["legend"]
                    ["train"], **line_kw_train)
            ax.plot(x, df[y_test],  label=t["legend"]["test"],  **line_kw_test)
            ax.set_title(f'{t["title"]} — {title_suffix}\n',
                         fontsize=24, fontweight="bold")
            ax.set_xlabel(t["x"], fontsize=24, fontweight="bold")
            ax.set_ylabel(y_label, fontsize=24, fontweight="bold")
            ax.tick_params(axis="both", labelsize=24)
            ax.legend(loc="best", fontsize=24, frameon=True, fancybox=True)
            ax.grid(True)
            if save_dir:
                fig.savefig(os.path.join(save_dir, fname),
                            bbox_inches="tight", dpi=300)
        else:
            missing = [c for c in (y_train, y_test) if c not in df.columns]
            print(
                f"[AVISO] No se graficó {title_suffix}: faltan columnas {missing}")

    plot_pair("train_mse", "test_mse", "MSE", t["y"]["mse"], "metric_mse.png")

    plot_pair("train_rmse", "test_rmse", "RMSE",
              t["y"]["rmse"], "metric_rmse.png")

    plot_pair("train_rrmse", "test_rrmse", "RRMSE",
              t["y"]["rrmse"], "metric_rrmse.png")

    plot_pair("train_cc", "test_cc", "CC", t["y"]["cc"], "metric_cc.png")

    if "ot_lambda_alpha" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, df["ot_lambda_alpha"], label=t["legend"]
                ["lambda"], **line_kw_lambda)
        ax.set_title("λ (OT)\n", fontsize=24, fontweight="bold")
        ax.set_xlabel(t["x"], fontsize=24, fontweight="bold")
        ax.set_ylabel(t["y"]["lambda"], fontsize=24, fontweight="bold")
        ax.tick_params(axis="both", labelsize=24)
        ax.legend(loc="best", fontsize=24, frameon=True, fancybox=True)
        ax.grid(True)
        if save_dir:
            fig.savefig(os.path.join(save_dir, "metric_lambda.png"),
                        bbox_inches="tight", dpi=300)
    else:
        print("[AVISO] No se graficó λ (OT): falta la columna 'ot_lambda_alpha'.")

    plt.show()


if __name__ == "__main__":
    plot_regression_metrics(
        "trainig_metrics.csv",
        language="en",
        save_dir="plots",
    )
