#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import yaml
from huggingface_hub import HfApi, hf_hub_download

from pipeline_lib.dataset_pin import (
    CHEMQUESTS_QA_SHA256,
    CHEMQUESTS_REVISION,
    DatasetIntegrityError,
    sha256_file,
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


FULL_TEXT_JSONL = "full_text.jsonl"
FULL_TEXT_DIR = "full_text"


def _norm_id(s: Any) -> str:
    s = str(s).strip()
    return (s.lstrip("0") or "0") if s.isdigit() else s


def normalise_text(raw: str) -> str:
    """The rule that reproduces the full_text.jsonl of all published runs byte for byte:
    universal newlines (CR and CRLF become LF) and no leading or trailing whitespace.
    At the pinned release this changes 17 of 151 documents: 16 end in one extra newline, and document 52
    holds a carriage return where the LaTeX ``\\right`` was decoded upstream as CR + "ight" (same length, so no span moves)."""
    return raw.replace("\r\n", "\n").replace("\r", "\n").strip()


def build_full_text_jsonl(txt_dir: str | Path, metadata_jsonl: str | Path, out_path: str | Path) -> Dict[str, Any]:
    """Build full_text.jsonl ({"corpus_id", "text"} per line, sorted by corpus_id as a string, compact JSON with
    ASCII escapes and ``\\/`` for "/", the layout of the earlier file) from full_text/<id>.txt and metadata.jsonl.
    The set of ids in the metadata and the set of text files must be the same. Returns a report."""
    txt_dir = Path(txt_dir)
    ids = []
    with open(metadata_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(str(json.loads(line)["corpus_id"]))
    files = {p.stem for p in txt_dir.glob("*.txt")}
    if len(set(ids)) != len(ids):
        raise DatasetIntegrityError("duplicate corpus_id in metadata.jsonl")
    if set(ids) != files:
        raise DatasetIntegrityError(f"metadata.jsonl and {txt_dir} disagree: only in metadata {sorted(set(ids) - files)[:5]}, "
                                    f"only as text file {sorted(files - set(ids))[:5]}")
    changed: Dict[str, List[str]] = {"crlf_or_cr_to_lf": [], "stripped_whitespace": []}
    lines = []
    for i in sorted(ids):
        raw = (txt_dir / f"{i}.txt").read_bytes().decode("utf-8")
        text = normalise_text(raw)
        if "\r" in raw:
            changed["crlf_or_cr_to_lf"].append(i)
        if raw.replace("\r\n", "\n").replace("\r", "\n") != text:
            changed["stripped_whitespace"].append(i)
        lines.append(json.dumps({"corpus_id": i, "text": text}, ensure_ascii=True, separators=(",", ":")).replace("/", "\\/") + "\n")
    Path(out_path).write_bytes("".join(lines).encode("utf-8"))
    return {"n_documents": len(ids), "changed_by_normalisation": changed}


def verify_spans(full_text_jsonl: str | Path, qa_jsonl: str | Path) -> int:
    """Every evidence span must be exactly ``text[start_index:end_index]`` of its document. Returns the number checked."""
    docs = {}
    with open(full_text_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                docs[_norm_id(r["corpus_id"])] = r["text"]
    n, bad = 0, []
    with open(qa_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            doc = docs.get(_norm_id(r["corpus_id"]))
            if doc is None or doc[int(r["start_index"]):int(r["end_index"])] != r["content"]:
                bad.append(i)
    if bad:
        raise DatasetIntegrityError(f"{len(bad)} of {n} evidence spans do not match full_text.jsonl (question rows {bad[:10]}...)")
    return n


def _fetch_full_text_dir(repo_id: str, revision: Optional[str], hf_token: Optional[str], outdir: Path) -> Path:
    """Download full_text/<id>.txt of the pinned revision (hf_hub_download, one file each) into ``outdir``/full_text."""
    names = [n for n in HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision, token=hf_token)
             if n.startswith(FULL_TEXT_DIR + "/") and n.endswith(".txt")]
    if not names:
        raise DatasetIntegrityError(f"{repo_id}@{revision} has no {FULL_TEXT_DIR}/*.txt files")
    dest = ensure_dir(outdir / FULL_TEXT_DIR)
    for n in names:
        cached = hf_hub_download(repo_id=repo_id, filename=n, repo_type="dataset", revision=revision, token=hf_token)
        shutil.copyfile(cached, dest / Path(n).name)
    return dest


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
    SHA256 of ``qa.jsonl`` can be checked against the pinned release. ``full_text.jsonl`` does not exist in that
    release: it is built from ``full_text/<id>.txt`` and ``metadata.jsonl`` and every span is checked. Existing
    files are kept when ``overwrite`` is false, but are verified all the same.
    """
    outdir = ensure_dir(output_dir)

    want_full_text = FULL_TEXT_JSONL in data_files
    if want_full_text:
        # the pinned release has no full_text.jsonl: it is built from full_text/<id>.txt + metadata.jsonl afterwards
        data_files = [f for f in data_files if f != FULL_TEXT_JSONL]
        for need in ("metadata.jsonl", "qa.jsonl"):
            if need not in data_files:
                data_files = data_files + [need]

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

    if want_full_text:
        target = outdir / FULL_TEXT_JSONL
        if target.exists() and not overwrite:
            logging.info(f"Skip (exists): {target}")
        else:
            txt_dir = _fetch_full_text_dir(repo_id, revision, hf_token, outdir)
            report = build_full_text_jsonl(txt_dir, outdir / "metadata.jsonl", target)
            report.update({"repo_id": repo_id, "revision": revision, "full_text_jsonl_sha256": sha256_file(target)})
            (outdir / "full_text_build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            logging.info(f"Built {target} from {report['n_documents']} text files")
        n = verify_spans(target, outdir / "qa.jsonl")
        logging.info(f"all {n} evidence spans match full_text.jsonl")


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
