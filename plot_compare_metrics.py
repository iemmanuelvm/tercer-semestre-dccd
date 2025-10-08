import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_merge(model_csv: Path, base_csv: Path) -> pd.DataFrame:
    df_m = pd.read_csv(model_csv)
    df_b = pd.read_csv(base_csv)

    df_m = df_m[df_m["snr_idx"].astype(str) != "ALL"].copy()
    df_b = df_b[df_b["snr_idx"].astype(str) != "ALL"].copy()
    df_m["snr_idx"] = df_m["snr_idx"].astype(int)
    df_b["snr_idx"] = df_b["snr_idx"].astype(int)

    merged = df_m.merge(df_b, on="snr_idx",
                        suffixes=("_denoised", "_baseline"))
    merged = merged.sort_values("snr_idx").reset_index(drop=True)
    return merged


def snr_db_axis(n_points: int) -> np.ndarray:
    return np.linspace(-5, 5, n_points)


def plot_one(merged, metric, ylabel, out_file, higher_is_better, noise_label, show_bands=False):
    x_db = snr_db_axis(len(merged))

    y_d = merged[f"{metric}_mean_denoised"].values
    e_d = merged[f"{metric}_std_denoised"].values
    y_b = merged[f"{metric}_mean_baseline"].values
    e_b = merged[f"{metric}_std_baseline"].values

    delta = (y_d - y_b) if higher_is_better else (y_b - y_d)

    fig, ax = plt.subplots(figsize=(9, 5))
    l1 = ax.errorbar(x_db, y_d, yerr=e_d, fmt="-o",
                     capsize=3, label="Denoised (Model)")
    l2 = ax.errorbar(x_db, y_b, yerr=e_b, fmt="--s",
                     capsize=3, label="Baseline (Noisy)")

    if show_bands:
        ax.fill_between(x_db, y_d - e_d, y_d + e_d, alpha=0.15)
        ax.fill_between(x_db, y_b - e_b, y_b + e_b, alpha=0.15)

    for xd, yd0, yb0, d in zip(x_db, y_d, y_b, delta):
        ax.plot([xd, xd], [yb0, yd0], linestyle=":", linewidth=1)
        ax.annotate(f"Δ={d:.3g}", xy=(xd, (yd0 + yb0) / 2),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=8)

    ax.set_xlim(-5, 5)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} vs. SNR (dB) — {noise_label}")
    ax.grid(True, alpha=0.6)

    ax.legend(handles=[l1.lines[0], l2.lines[0]], labels=[
              "Denoised (Model)", "Baseline (Noisy)"])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Compare metrics vs SNR for denoised vs baseline.")
    ap.add_argument("--noise", default="CHEW",
                    help="Noise tag used in file names (e.g., CHEW)")
    ap.add_argument("--in-dir", default=".",
                    help="Directory containing metrics_*.csv files")
    ap.add_argument("--out-dir", default=".", help="Directory to save figures")
    ap.add_argument("--bands", action="store_true",
                    help="Show mean ± std shaded bands")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    noise = args.noise.upper()

    model_csv = in_dir / f"metrics_{noise}.csv"
    base_csv = in_dir / f"metrics_{noise}_BASELINE.csv"
    if not model_csv.exists() or not base_csv.exists():
        raise FileNotFoundError(f"Missing CSVs:\n- {model_csv}\n- {base_csv}")

    merged = load_and_merge(model_csv, base_csv)

    # Plot all metrics
    plots = [
        ("CC", "Correlation Coefficient (CC)", True),
        ("RMSE", "RMSE", False),
        ("RRMSE", "RRMSE", False),
        ("MSE", "MSE", False),
    ]
    for metric, ylabel, hib in plots:
        out_file = out_dir / f"{noise.lower()}_compare_{metric}_db_xlim.png"
        plot_one(
            merged=merged,
            metric=metric,
            ylabel=ylabel,
            out_file=out_file,
            higher_is_better=hib,
            noise_label=noise,
            show_bands=args.bands,
        )
        print(f"[OK] Saved -> {out_file}")


if __name__ == "__main__":
    main()


# python plot_compare_metrics.py --noise CHEW --in-dir . --out-dir ./figs_compare_baseline_noise_removal
# python plot_compare_metrics.py --noise ELPP --in-dir . --out-dir ./figs_compare_baseline_noise_removal
# python plot_compare_metrics.py --noise EMG --in-dir . --out-dir ./figs_compare_baseline_noise_removal
# python plot_compare_metrics.py --noise EOG --in-dir . --out-dir ./figs_compare_baseline_noise_removal
# python plot_compare_metrics.py --noise SHIV --in-dir . --out-dir ./figs_compare_baseline_noise_removal
