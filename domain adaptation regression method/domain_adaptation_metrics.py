import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


def plot_divergence_metrics(
    csv_path: str,
    language: str = "es",
    save_dir: str | None = None,
):
    plt.rc("font", family="Times New Roman")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")

    num_cols = [
        "train_KL", "train_JSD", "train_JSDist",
        "test_KL",  "test_JSD",  "test_JSDist",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            raise ValueError(f"Falta la columna requerida: '{c}' en el CSV.")

    texts = {
        "es": {
            "title": "Entrenamiento vs Prueba",
            "x": "Épocas",
            "y": {
                "kl": "Divergencia KL",
                "jsd": "Jensen–Shannon Divergence (JSD)",
                "jsdist": "Jensen–Shannon Distance (JSDist)",
            },
            "legend": {"train": "Entrenamiento", "test": "Prueba"},
        },
        "en": {
            "title": "Training vs Test",
            "x": "Epochs",
            "y": {
                "kl": "KL Divergence",
                "jsd": "Jensen–Shannon Divergence (JSD)",
                "jsdist": "Jensen–Shannon Distance (JSDist)",
            },
            "legend": {"train": "Train", "test": "Test"},
        },
    }
    t = texts["es" if language == "es" else "en"]

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    line_kw_train = dict(linewidth=1, linestyle="-", marker="x", markersize=5)
    line_kw_test = dict(linewidth=1, linestyle="--", marker="s", markersize=5)

    x = df["epoch"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, df["train_KL"], label=t["legend"]["train"], **line_kw_train)
    ax.plot(x, df["test_KL"],  label=t["legend"]["test"],  **line_kw_test)
    ax.set_title(f'{t["title"]} — KL\n', fontsize=18, fontweight="bold")
    ax.set_xlabel(t["x"], fontsize=18, fontweight="bold")
    ax.set_ylabel(t["y"]["kl"], fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14, frameon=True, fancybox=True)
    ax.grid(True)
    if save_dir:
        fig.savefig(os.path.join(save_dir, "metric_KL.png"),
                    bbox_inches="tight", dpi=300)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, df["train_JSD"], label=t["legend"]["train"], **line_kw_train)
    ax.plot(x, df["test_JSD"],  label=t["legend"]["test"],  **line_kw_test)
    ax.set_title(f'{t["title"]} — JSD\n', fontsize=18, fontweight="bold")
    ax.set_xlabel(t["x"], fontsize=18, fontweight="bold")
    ax.set_ylabel(t["y"]["jsd"], fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14, frameon=True, fancybox=True)
    ax.grid(True)
    if save_dir:
        fig.savefig(os.path.join(save_dir, "metric_JSD.png"),
                    bbox_inches="tight", dpi=300)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, df["train_JSDist"], label=t["legend"]["train"], **line_kw_train)
    ax.plot(x, df["test_JSDist"],  label=t["legend"]["test"],  **line_kw_test)
    ax.set_title(f'{t["title"]} — JSDist\n', fontsize=18, fontweight="bold")
    ax.set_xlabel(t["x"], fontsize=18, fontweight="bold")
    ax.set_ylabel(t["y"]["jsdist"], fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14, frameon=True, fancybox=True)
    ax.grid(True)
    if save_dir:
        fig.savefig(os.path.join(save_dir, "metric_JSDist.png"),
                    bbox_inches="tight", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_divergence_metrics(
        "trainig_metrics.csv",
        language="en",
        save_dir="plots",
    )
