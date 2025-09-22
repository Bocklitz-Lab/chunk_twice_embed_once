#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from datasets import load_dataset


def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}



def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def export_jsonl(dataset, out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        logging.info(f"Skip (exists): {out_path}")
        return
    # datasets.Dataset.to_json writes JSON Lines (one obj per line)
    logging.info(f"Writing: {out_path}")
    dataset.to_json(str(out_path), orient="records", lines=True)


def download_files(
    repo_id: str,
    data_files: List[str],
    output_dir: str,
    hf_token: Optional[str] = None,
    revision: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    outdir = ensure_dir(output_dir)

    for file in data_files:
        logging.info(f"Loading {repo_id}:{file} (rev={revision or 'main'})")
        ds_dict = load_dataset(
            repo_id,
            data_files=file,
            split=None,          # default; returns a DatasetDict with 'train'
            token=hf_token,
            revision=revision,
        )
        ds = ds_dict["train"]

        # Write out with the same filename
        out_path = outdir / Path(file).name
        export_jsonl(ds, out_path, overwrite)


def main():
    parser = argparse.ArgumentParser(
        description="Download JSONL artifacts from a HF dataset repo based on a config file."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config with parameters (see example).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)."
    )
    args = parser.parse_args()
    setup_logger(args.verbose)

    cfg = read_config(args.config)

    repo_id = cfg.get("repo_id", "Bocklitz-Lab/ChemQuests")
    data_files = cfg.get(
        "data_files",
        ["metadata.jsonl", "full_text.jsonl", "qa.jsonl"]
    )
    output_dir = cfg.get("output_dir", "./chemquests_artifacts")
    hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN")
    revision = cfg.get("revision")  # e.g., "main" or a specific commit/branch/tag
    overwrite = bool(cfg.get("overwrite", False))

    logging.info(f"Repo: {repo_id}")
    logging.info(f"Files: {data_files}")
    logging.info(f"Output: {output_dir}")
    logging.info(f"Revision: {revision or 'main'}")
    logging.info(f"Overwrite: {overwrite}")

    download_files(
        repo_id=repo_id,
        data_files=data_files,
        output_dir=output_dir,
        hf_token=hf_token,
        revision=revision,
        overwrite=overwrite,
    )

    print(f"Done. Artifacts are in: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
