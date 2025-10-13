#!/usr/bin/env python3
import argparse
import pandas as pd
import os
import sys


def merge_loss_and_runs(loss_csv: str, runs_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Merge loss.csv (with 'rel_path') and merged_runs.csv (with 'outdir')
    by extracting a common key and joining them.
    """

    # --- Read CSVs ---
    df1 = pd.read_csv(loss_csv)
    df2 = pd.read_csv(runs_csv)

    if "rel_path" not in df1.columns:
        sys.exit("❌ 'rel_path' column not found in first CSV.")
    if "outdir" not in df2.columns:
        sys.exit("❌ 'outdir' column not found in second CSV.")

    # --- Extract merge keys ---
    df1["merge_key"] = df1["rel_path"].str.extract(r"([^/]+)/results")
    df2["merge_key"] = df2["outdir"].str.extract(r"artifacts/tests/([^/]+)")

    # --- Merge ---
    merged_df = pd.merge(df1, df2, on="merge_key", how="inner")

    # --- Cleanup ---
    merged_df.drop(columns=["merge_key"], inplace=True)

    # --- Save ---
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    merged_df.to_csv(output_csv, index=False)

    return merged_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge loss.csv and merged_runs.csv on matching run IDs."
    )
    parser.add_argument(
        "--loss", dest="loss_csv", required=True,
        help="Path to loss.csv (contains 'rel_path' column)"
    )
    parser.add_argument(
        "--runs", dest="runs_csv", required=True,
        help="Path to merged_runs.csv (contains 'outdir' column)"
    )
    parser.add_argument(
        "--out", dest="output_csv", default="../artifacts/combined.csv",
        help="Path to save merged CSV (default: ../artifacts/combined.csv)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merged = merge_loss_and_runs(args.loss_csv, args.runs_csv, args.output_csv)

    print("✅ Merged dataframe saved to:", args.output_csv)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(merged.head(10))
