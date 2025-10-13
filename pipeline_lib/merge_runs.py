# %% [markdown]
# ## 1. Imports & Constants
# Load required libraries and define input/output paths.

# %%
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Defaults (can be overridden by CLI args)
DEFAULT_CSV = "../artifacts/tests/durations.csv"
DEFAULT_SAVE = "../artifacts/tests/merged_runs.csv"
DEFAULT_TZ = "Europe/Berlin"
OUTDIR = "figures"  # (currently unused)

# --- Safe fallbacks for model mapping helpers (in case they aren't defined elsewhere) ---
def _norm_model_key(name: str) -> str:
    """Normalize model name for mapping keys."""
    if not isinstance(name, str):
        return ""
    return name.strip().lower().replace(" ", "").replace(":", "-")

# If you already define `model_name_map` elsewhere, this local definition will be shadowed.
model_name_map = {
    # "gpt-4o-mini-2024-07-18": "g4o-mini",
    # "gpt-4o": "g4o",
    # Add any custom mappings you want; empty means identity.
}


def merge_runs_by_outdir(csv_path: str,
                         tz: str | None = DEFAULT_TZ,
                         save_path: str | None = DEFAULT_SAVE) -> pd.DataFrame:
    """
    Read the CSV of stage runs and merge triplets (stage4/5/6) by `outdir`.
    - Keeps per-stage started/ended/duration/status.
    - Keeps shared metadata (model, revision, chunker, size, overlap) once.
    - Adds `duration_sum_s` = sum of available stage durations.
    - Timezone-aware parsing from the 'Z' timestamps; converts to `tz` if given.

    Returns the merged DataFrame and (optionally) writes it to CSV.
    """

    # --- Read & parse ---
    df = pd.read_csv(
        csv_path,
        dtype={
            "duration_s": "Int64",
            "status": "Int64",
            "stage": "string",
            "model": "string",
            "revision": "string",
            "chunker": "string",
            "size": "Int64",
            "overlap": "Int64",
            "outdir": "string",
        }
    )

    # Map to a shorter model label if a mapping is available; otherwise keep as-is
    df["model_short"] = (
        df["model"].astype("string")
        .map(lambda x: model_name_map.get(_norm_model_key(str(x)), x))
        .astype("string")
    )

    # Parse times as UTC and optionally convert to desired tz
    for col in ["started_at", "ended_at"]:
        dt = pd.to_datetime(df[col], utc=True, errors="coerce")
        if tz:
            dt = dt.dt.tz_convert(tz)
        df[col] = dt

    # --- Pivot per-stage fields ---
    per_stage_cols = ["started_at", "ended_at", "duration_s", "status"]
    wide = (
        df.set_index(["outdir", "stage"])[per_stage_cols]
          .unstack("stage")  # columns become a MultiIndex (field, stage)
    )

    # Flatten MultiIndex column names: e.g., ('duration_s','stage4') -> 'duration_s_stage4'
    wide.columns = [f"{field}_{stage}" for field, stage in wide.columns]

    # --- Bring along shared metadata (assumed identical within an outdir) ---
    shared_cols = ["model", "model_short", "revision", "chunker", "size", "overlap"]
    meta = (
        df.drop_duplicates(subset=["outdir"])[["outdir"] + shared_cols]
          .set_index("outdir")
    )

    merged = meta.join(wide, how="left").reset_index()

    # --- Add total duration across available stages ---
    duration_cols = [c for c in merged.columns if c.startswith("duration_s_stage")]
    merged["duration_sum_s"] = merged[duration_cols].sum(axis=1, skipna=True).astype("Int64")

    # Optional: sort columns nicely
    def order_cols(cols):
        base = ["outdir"] + shared_cols
        stages = ["stage4", "stage5", "stage6"]
        per = ["started_at", "ended_at", "duration_s", "status"]

        ordered = base[:]
        for st in stages:
            for p in per:
                name = f"{p}_{st}"
                if name in cols:
                    ordered.append(name)
        if "duration_sum_s" in cols:
            ordered.append("duration_sum_s")
        ordered += [c for c in cols if c not in ordered]
        return ordered

    merged = merged[order_cols(list(merged.columns))]

    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        merged.to_csv(save_path, index=False)

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge stage runs (4/5/6) by outdir and compute total durations."
    )
    parser.add_argument(
        "--csv", "--csv_path", dest="csv_path", default=DEFAULT_CSV,
        help=f"Path to the input CSV (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--tz", dest="tz", default=DEFAULT_TZ,
        help=f"IANA timezone to convert timestamps to, or empty to keep UTC (default: {DEFAULT_TZ})"
    )
    parser.add_argument(
        "--save", "--save_path", dest="save_path", default=DEFAULT_SAVE,
        help=f"Path to write merged CSV, or empty to skip saving (default: {DEFAULT_SAVE})"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write the merged CSV to disk (overrides --save/--save_path)."
    )
    return parser.parse_args()


# --- CLI entrypoint ---
if __name__ == "__main__":
    args = parse_args()
    save_path = None if args.no_save or (args.save_path is not None and args.save_path.strip() == "") else args.save_path
    tz = None if args.tz is not None and args.tz.strip() == "" else args.tz

    merged_df = merge_runs_by_outdir(
        csv_path=args.csv_path,
        tz=tz,
        save_path=save_path
    )

    # Print a quick preview to stdout
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(merged_df.head(10))
        cols = [c for c in merged_df.columns if c.startswith("duration_s_stage")]
        check = merged_df[["outdir", "duration_sum_s"] + cols]
        print("\n[Duration check]")
        print(check.head(10))
