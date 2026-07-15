#!/usr/bin/env python
"""Report CLI — gom moi runs/*_metrics.json (do eval.py sinh ra) thanh 1 bang.

Vi du:
    !python report.py                        # doc runs/*_metrics.json -> runs/metrics_summary.csv
    !python report.py --out-dir runs --out-csv runs/summary.csv

Khong train, khong eval — chi tong hop de so sanh.
"""
import argparse
import glob
import json
import os

import pandas as pd

SUFFIX = "_metrics.json"
SHOW = ["model", "em_overall", "em_closed", "em_open",
        "token_f1_open", "sem_open",
        "clin_balanced_acc", "clin_auc", "params_trainable"]


def main():
    p = argparse.ArgumentParser(description="Gom metrics json thanh bang so sanh")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.out_dir, "*" + SUFFIX))):
        name = os.path.basename(path)[:-len(SUFFIX)]
        with open(path) as f:
            m = json.load(f)
        rows.append({"model": name, **m})

    if not rows:
        print(f"⚠️ Khong tim thay *{SUFFIX} trong {args.out_dir}/ "
              f"(chay eval.py voi --save_metrics truoc)")
        return

    df = pd.DataFrame(rows)
    if "em_overall" in df.columns:
        df = df.sort_values("em_overall", ascending=False)
    out_csv = args.out_csv or os.path.join(args.out_dir, "metrics_summary.csv")
    df.to_csv(out_csv, index=False)   # CSV giu full precision + ca run smoke

    # Bang so sanh in ra: bo run smoke (1 epoch, degenerate) + format 6 chu so thap phan
    show_df = df[~df["model"].str.startswith("smoke")] if "model" in df.columns else df
    cols = [c for c in SHOW if c in show_df.columns]
    print(show_df[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\n💾 Da luu: {out_csv}")


if __name__ == "__main__":
    main()
