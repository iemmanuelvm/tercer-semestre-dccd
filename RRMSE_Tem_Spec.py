import os
import argparse
import glob
import pandas as pd
import numpy as np


def read_all_row(df: pd.DataFrame) -> pd.Series:
    if "snr_idx" not in df.columns:
        raise ValueError("snr_idx column not found")
    s = df.copy()
    s["snr_idx_str"] = s["snr_idx"].astype(str).str.upper()
    all_rows = s[s["snr_idx_str"] == "ALL"]
    if len(all_rows) == 1:
        return all_rows.iloc[0]
    cols = [c for c in df.columns if c.endswith("_mean")]
    rr = {}
    for c in cols:
        rr[c.replace("_mean", "_overall")] = float(
            s.loc[s["snr_idx_str"] != "ALL", c].astype(float).mean())
    base = s.iloc[0].copy()
    for k, v in rr.items():
        base[k] = v
    base["snr_idx"] = "ALL"
    return base


def extract_metrics(row: pd.Series, file_name: str) -> dict:
    cols = row.index.tolist()
    def get(k): return (
        float(row[k]) if k in cols and pd.notna(row[k]) else np.nan)
    noise_val = row["noise"] if "noise" in cols else os.path.splitext(
        os.path.basename(file_name))[0]
    return {
        "file": os.path.basename(file_name),
        "noise": str(noise_val),
        "CC_overall": get("CC_overall"),
        "MSE_overall": get("MSE_overall"),
        "RMSE_overall": get("RMSE_overall"),
        "RRMSE_overall": get("RRMSE_overall"),
        "RRMSE_temp_overall": get("RRMSE_temp_overall"),
        "RRMSE_spec_overall": get("RRMSE_spec_overall"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./inferences")
    ap.add_argument("--pattern", default="metrics_*.csv")
    ap.add_argument("--out", default="rrmse_summary.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print("No CSV files found")
        return

    rows = []
    for p in paths:
        try:
            df = pd.read_csv(p, dtype={"snr_idx": str})
            all_row = read_all_row(df)
            rows.append(extract_metrics(all_row, p))
        except Exception as e:
            rows.append({"file": os.path.basename(p), "noise": os.path.basename(p), "CC_overall": np.nan, "MSE_overall": np.nan,
                        "RMSE_overall": np.nan, "RRMSE_overall": np.nan, "RRMSE_temp_overall": np.nan, "RRMSE_spec_overall": np.nan})

    out_df = pd.DataFrame(rows)
    cols_order = ["file", "noise", "CC_overall", "MSE_overall", "RMSE_overall",
                  "RRMSE_overall", "RRMSE_temp_overall", "RRMSE_spec_overall"]
    out_df = out_df[cols_order]
    out_path = os.path.join(args.dir, args.out) if not os.path.isabs(
        args.out) else args.out
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()


# python RRMSE_Tem_Spec.py --dir "./inferences" --pattern "metrics_CHEW.csv"
# python RRMSE_Tem_Spec.py --dir "./inferences" --pattern "metrics_ELPP.csv"
# python RRMSE_Tem_Spec.py --dir "./inferences" --pattern "metrics_EMG.csv"
# python RRMSE_Tem_Spec.py --dir "./inferences" --pattern "metrics_EOG.csv"
# python RRMSE_Tem_Spec.py --dir "./inferences" --pattern "metrics_SHIV.csv"
