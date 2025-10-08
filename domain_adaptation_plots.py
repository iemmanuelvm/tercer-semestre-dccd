import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


def plot_train_metrics_one(
    csv_path: str,
    language: str = "es",
    save_dir: str | None = None,
):
    plt.rc("font", family="Times New Roman")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")

    required = ["train_KL", "train_JSD", "train_JSDist"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Falta la columna requerida: '{c}' en el CSV.")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    texts = {
        "es": {
            "title": "Métricas de Divergencia",
            "x": "Épocas",
            "y": "Valores",
            "legend": {
                "kl": "Divergencia KL Promedio",
                "jsd": "Divergencia Jensen–Shannon Promedio",
                "jsdist": "Distancia Jensen–Shannon Promedio",
            },
            "fname": "metricas_entrenamiento.png",
        },
        "en": {
            "title": "Divergence Metrics (Training)",
            "x": "Epochs",
            "y": "Values",
            "legend": {
                "kl": "Average KL Divergence",
                "jsd": "Average Jensen–Shannon Divergence",
                "jsdist": "Average Jensen–Shannon Distance",
            },
            "fname": "training_metrics.png",
        },
    }
    t = texts["es" if language == "es" else "en"]

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    x = df["epoch"]
    fig, ax = plt.subplots(figsize=(10, 6))

    style = dict(linewidth=1.5, markersize=5)
    ax.plot(x, df["train_KL"],      ":o",
            label=t["legend"]["kl"],     **style)
    ax.plot(x, df["train_JSD"],     ":^",
            label=t["legend"]["jsd"],    **style)
    ax.plot(x, df["train_JSDist"],  ":<",
            label=t["legend"]["jsdist"], **style)

    ax.set_title(t["title"], fontsize=24, fontweight="bold")
    ax.set_xlabel(t["x"], fontsize=24, fontweight="bold")
    ax.set_ylabel(t["y"], fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", labelsize=24)
    ax.legend(loc="best", fontsize=24, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        fig.savefig(os.path.join(
            save_dir, t["fname"]), bbox_inches="tight", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_train_metrics_one(
        "trainig_metrics.csv",
        language="en",
        save_dir="plots",
    )
