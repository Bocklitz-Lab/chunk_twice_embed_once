"""Stage 5 entry point: build the MTEB task with the label method named in the config.

``labeling.method`` of the stage 5 config decides which script runs:
  span     -> stages/make_span_qrels.py  (labels from the evidence-span positions, cover level c)
  lexical  -> stages/make_mteb_task.py   (BM25 + Jaccard + fuzzy labels; also written as qrels_lexical.jsonl)
A config without a ``labeling`` block keeps the lexical labels (configs generated before the span labels existed).
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib.span_labels import METHODS  # noqa: E402


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            import yaml  # PyYAML

            return yaml.safe_load(f) or {}
        return json.load(f)


def label_method(cfg: Dict[str, Any]) -> str:
    method = (cfg.get("labeling") or {}).get("method", "lexical")
    if method not in METHODS:
        raise SystemExit(f"ERROR: labeling.method must be one of {METHODS}, got {method!r}")
    return method


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 5: MTEB task (queries, corpus, qrels) for the configured label method.")
    ap.add_argument("--config", required=True, help="stage 5 config (YAML/JSON)")
    args = ap.parse_args(argv)
    method = label_method(load_config(args.config))
    print(f"▶️ Stage5 label method: {method}")
    if method == "span":
        from stages.make_span_qrels import main as span_main

        return span_main(["--config", args.config])
    # make_mteb_task.py is a script (its logic runs at import); run it as __main__ with the same --config
    saved = sys.argv
    sys.argv = ["make_mteb_task.py", "--config", args.config]
    try:
        runpy.run_module("stages.make_mteb_task", run_name="__main__")
    finally:
        sys.argv = saved
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
