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
    return merged.sort_values("snr_idx").reset_index(drop=True)


def snr_db_axis(n_points: int) -> np.ndarray:
    return np.linspace(-5, 5, n_points)


def plot_one(
    merged: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_file: Path,
    higher_is_better: bool,
    noise_label: str,
    show_bands: bool = False,
    percent_labels: bool = False,
):
    x_db = snr_db_axis(len(merged))
    y_d = merged[f"{metric}_mean_denoised"].values
    e_d = merged[f"{metric}_std_denoised"].values
    y_b = merged[f"{metric}_mean_baseline"].values
    e_b = merged[f"{metric}_std_baseline"].values

    if percent_labels:
        if higher_is_better:
            ann = 100.0 * (y_d - y_b) / np.maximum(y_b, 1e-12)
            ann_text = "% increase"
        else:
            ann = 100.0 * (y_b - y_d) / np.maximum(y_b, 1e-12)
            ann_text = "% reduction"
    else:
        ann = (y_d - y_b) if higher_is_better else (y_b - y_d)
        ann_text = "Δ"

    fig, ax = plt.subplots(figsize=(9, 5))
    l1 = ax.errorbar(x_db, y_d, yerr=e_d, fmt="-o",
                     capsize=3, label="Denoised (Model)")
    l2 = ax.errorbar(x_db, y_b, yerr=e_b, fmt="--s",
                     capsize=3, label="Baseline (Noisy)")

    if show_bands:
        ax.fill_between(x_db, y_d - e_d, y_d + e_d, alpha=0.15)
        ax.fill_between(x_db, y_b - e_b, y_b + e_b, alpha=0.15)

    for xd, yd0, yb0, a in zip(x_db, y_d, y_b, ann):
        ax.plot([xd, xd], [yb0, yd0], linestyle=":", linewidth=1)
        ax.annotate(f"{a:.1f}%" if percent_labels else f"Δ={a:.3g}",
                    xy=(xd, (yd0 + yb0) / 2), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlim(-5, 5)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    title_suffix = f"({ann_text} annotated)" if percent_labels else ""
    ax.set_title(f"{metric} vs. SNR (dB) — {noise_label} {title_suffix}")
    ax.grid(True, alpha=0.6)
    ax.legend(handles=[l1.lines[0], l2.lines[0]], labels=[
              "Denoised (Model)", "Baseline (Noisy)"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def per_snr_percent_improvement(merged: pd.DataFrame, metric: str, higher_is_better: bool) -> np.ndarray:
    y_d = merged[f"{metric}_mean_denoised"].values
    y_b = merged[f"{metric}_mean_baseline"].values
    if higher_is_better:
        return 100.0 * (y_d - y_b) / np.maximum(y_b, 1e-12)
    else:
        return 100.0 * (y_b - y_d) / np.maximum(y_b, 1e-12)


def main():
    ap = argparse.ArgumentParser(
        description="General comparison: denoised vs baseline across noises.")
    ap.add_argument("--noises", default="CHEW,ELPP,EMG,EOG,SHIV",
                    help="Comma-separated noise list, e.g. CHEW,ELPP,EMG,EOG,SHIV")
    ap.add_argument("--in-dir", default=".",
                    help="Directory with metrics_*.csv files")
    ap.add_argument("--out-dir", default="./figs_compare_baseline_noise_removal",
                    help="Output directory for figures")
    ap.add_argument("--bands", action="store_true",
                    help="Show mean ± std shaded bands")
    ap.add_argument("--percent", action="store_true",
                    help="Annotate % improvement instead of Δ")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    noises = [n.strip().upper() for n in args.noises.split(",") if n.strip()]
    metrics_cfg = [
        ("CC", "Correlation Coefficient (CC)", True),
        ("RMSE", "RMSE", False),
        ("RRMSE", "RRMSE", False),
        ("MSE", "MSE", False),
    ]

    summary_rows = []
    for noise in noises:
        model_csv = in_dir / f"metrics_{noise}.csv"
        base_csv = in_dir / f"metrics_{noise}_BASELINE.csv"
        if not model_csv.exists() or not base_csv.exists():
            print(f"[WARN] Missing CSVs for {noise}. Skipping.")
            continue

        merged = load_and_merge(model_csv, base_csv)
        x_db = snr_db_axis(len(merged))

        for metric, ylabel, hib in metrics_cfg:
            out_file = out_dir / \
                f"{noise.lower()}_compare_{metric}_db_xlim.png"
            plot_one(merged, metric, ylabel, out_file, hib, noise,
                     show_bands=args.bands, percent_labels=args.percent)
            print(f"[OK] Saved -> {out_file}")

            pct = per_snr_percent_improvement(merged, metric, hib)
            for xd, p in zip(x_db, pct):
                summary_rows.append({"noise": noise, "snr_db": float(
                    xd), "metric": metric, "pct_improvement": float(p)})

    if not summary_rows:
        print("[ERROR] No data summarised. Check --in-dir and noise names.")
        return

    summary = pd.DataFrame(summary_rows)
    csv_summary = out_dir / "summary_percent_improvement.csv"
    summary.to_csv(csv_summary, index=False)
    print(f"[OK] Summary CSV -> {csv_summary}")

    for metric, ylabel, hib in metrics_cfg:
        fig, ax = plt.subplots(figsize=(9, 5))
        for noise in noises:
            df_n = summary[(summary["noise"] == noise) & (
                summary["metric"] == metric)].sort_values("snr_db")
            if df_n.empty:
                continue
            ax.plot(df_n["snr_db"], df_n["pct_improvement"],
                    marker="o", label=noise)
        ax.axhline(0, color="k", linewidth=1, alpha=0.5)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("% improvement" if hib else "% reduction")
        ax.set_title(f"{metric}: % improvement vs SNR across noises")
        ax.set_xlim(-5, 5)
        ax.set_xticks(np.arange(-5, 6, 1))
        ax.grid(True, alpha=0.6)
        ax.legend(ncol=2)
        out_file = out_dir / f"all_noises_{metric}_pct_vs_snr.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved -> {out_file}")

    for metric, ylabel, hib in metrics_cfg:
        dfm = summary[summary["metric"] == metric].groupby("noise", as_index=False)[
            "pct_improvement"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(dfm["noise"], dfm["pct_improvement"])
        ax.set_ylabel("Mean % improvement" if hib else "Mean % reduction")
        ax.set_title(f"{metric}: mean % improvement across SNR (per noise)")
        ax.grid(axis="y", alpha=0.4)
        out_file = out_dir / f"all_noises_{metric}_mean_pct_bar.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved -> {out_file}")


if __name__ == "__main__":
    main()
