#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import yaml
from huggingface_hub import hf_hub_download

from pipeline_lib.dataset_pin import (
    CHEMQUESTS_QA_SHA256,
    CHEMQUESTS_REVISION,
    DatasetIntegrityError,
    verify_qa_sha256,
)


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


def download_files(
    repo_id: str,
    data_files: List[str],
    output_dir: str,
    hf_token: Optional[str] = None,
    revision: Optional[str] = None,
    overwrite: bool = False,
    expected_qa_sha256: Optional[str] = CHEMQUESTS_QA_SHA256,
) -> None:
    """Copy the raw files of a pinned dataset revision into ``output_dir``.

    Files are copied byte for byte (no re-export through ``datasets``), so the
    SHA256 of ``qa.jsonl`` can be checked against the pinned release. Existing
    files are kept when ``overwrite`` is false, but are verified all the same.
    """
    outdir = ensure_dir(output_dir)

    for file in data_files:
        out_path = outdir / Path(file).name
        if out_path.exists() and not overwrite:
            logging.info(f"Skip (exists): {out_path}")
        else:
            logging.info(f"Downloading {repo_id}:{file} (rev={revision})")
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                repo_type="dataset",
                revision=revision,
                token=hf_token,
            )
            shutil.copyfile(cached, out_path)
            logging.info(f"Wrote: {out_path}")

        if Path(file).name == "qa.jsonl" and expected_qa_sha256:
            digest = verify_qa_sha256(out_path, expected_qa_sha256)
            logging.info(f"qa.jsonl SHA256 OK: {digest}")


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
    revision = cfg.get("revision", CHEMQUESTS_REVISION)  # pinned commit; never a moving ref
    if revision in (None, "", "main", "master"):
        raise DatasetIntegrityError(
            f"Dataset revision must be a pinned commit hash, got {revision!r} "
            f"(expected {CHEMQUESTS_REVISION})."
        )
    expected_qa_sha256 = cfg.get("qa_sha256") or CHEMQUESTS_QA_SHA256
    overwrite = bool(cfg.get("overwrite", False))

    logging.info(f"Repo: {repo_id}")
    logging.info(f"Files: {data_files}")
    logging.info(f"Output: {output_dir}")
    logging.info(f"Revision: {revision}")
    logging.info(f"Overwrite: {overwrite}")

    download_files(
        repo_id=repo_id,
        data_files=data_files,
        output_dir=output_dir,
        hf_token=hf_token,
        revision=revision,
        overwrite=overwrite,
        expected_qa_sha256=expected_qa_sha256,
    )

    print(f"Done. Artifacts are in: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
