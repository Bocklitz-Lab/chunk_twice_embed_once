
# # run_eval.py
# from sentence_transformers import SentenceTransformer
# from mteb import MTEB
# from my_tasks import ChemQuest
# from model_wrapper import STNoPrompt

# base = SentenceTransformer("all-MiniLM-L6-v2")
# model = STNoPrompt(base, normalize=True)  # simple pass-through, no task prompts

# task = ChemQuest()            # or pass the class: MTEB(tasks=[ChemQuest])
# evaluation = MTEB(tasks=[task])

# results = evaluation.run(
#     model,
#     model_name="all-MiniLM-L6-v2",
#     model_revision="main",          # or pin a commit hash
#     output_folder="1_ChemQuest",
# )

# print(results)




from __future__ import annotations
import argparse
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

# 3rd-party
from sentence_transformers import SentenceTransformer
from mteb import MTEB, get_tasks

# local
from pipeline_lib.model_wrapper import STNoPrompt  # keep your wrapper
from pipeline_lib.prompts import resolve_prompts
from tools import metrics as qmetrics
from pipeline_lib.run_manifest import (
    Stopwatch, build_manifest, set_seeds, summarize_encode_timings, write_manifest,
)

# =================== CONFIG LOADING =========================================
def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML/JSON/TOML config based on file extension.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # PyYAML
        except ImportError as e:
            raise RuntimeError("PyYAML is required for YAML config files. `pip install pyyaml`") from e
        return yaml.safe_load(text) or {}
    elif suffix == ".json":
        return json.loads(text or "{}")
    elif suffix == ".toml":
        try:
            import tomllib  # py3.11+
        except Exception:
            try:
                import tomli as tomllib  # py3.10-
            except ImportError as e:
                raise RuntimeError("TOML requires Python 3.11+ or `pip install tomli`.") from e
        return tomllib.loads(text or "")
    else:
        raise ValueError(f"Unsupported config format: {suffix}. Use .yaml/.yml, .json, or .toml")

# =================== CONFIG SCHEMA ==========================================
@dataclass
class ModelCfg:
    name: str
    revision: str = "main"
    normalize: bool = True
    batch_size: int = 64
    # Future: allow custom encode kwargs per model
    encode_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    # optional prompt entry for this model only ({query, passage}); wins over the model_prompts table
    prompts: Optional[Dict[str, Any]] = None
@dataclass
class TaskSpec:
    # One of:
    #   - official task name (string) e.g. "ChemNQRetrieval"
    #   - import path e.g. "my_tasks.ChemQuest"
    #   - dict: {"class": "my_tasks.ChemQuest", "kwargs": {...}}
    spec: Union[str, Dict[str, Any]]

@dataclass
class AppConfig:
    models: List[ModelCfg]
    tasks: List[TaskSpec]
    output_root: Path = Path("1_MultiRetrieval")
    # Optional: modules to pre-import (useful if tasks given by simple class name)
    import_modules: List[str] = field(default_factory=list)

    # Optional global encode kwargs (merged with model.encode_kwargs, model wins)
    encode_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Optional retrieval depth passed to MTEB (top_k). None = MTEB's default (1000), which is what
    # all published runs used. A smaller value (e.g. 100) only truncates the saved rankings and the
    # @1000 metrics; metrics at k <= top_k are unchanged.
    top_k: Optional[int] = None

    # Prompts (see pipeline_lib/prompts.py). Default False: a config written before prompts existed keeps the
    # old behaviour (no prompt). The generated stage files carry params.use_prompts / params.model_prompts.
    use_prompts: bool = False
    model_prompts: Dict[str, Any] = field(default_factory=dict)

    # Seed for python / numpy / torch (null = not seeded). Written to the run manifest.
    seed: Optional[int] = None

def parse_app_config(raw: Dict[str, Any]) -> AppConfig:
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping.")

    # models
    models_raw = raw.get("models")
    if not models_raw or not isinstance(models_raw, list):
        raise ValueError("Config must define a non-empty list `models`.")

    models: List[ModelCfg] = []
    for i, m in enumerate(models_raw):
        if not isinstance(m, dict):
            raise ValueError(f"`models[{i}]` must be a mapping.")
        name = m.get("name")
        if not name:
            raise ValueError(f"`models[{i}].name` is required.")
        models.append(
            ModelCfg(
                name=name,
                revision=m.get("revision", "main"),
                normalize=bool(m.get("normalize", True)),
                batch_size=int(m.get("batch_size", 64)),
                encode_kwargs=m.get("encode_kwargs", {}) or {},
                model_kwargs=m.get("model_kwargs", {}) or {},
                prompts=m.get("prompts"),
            )
        )

    # tasks
    tasks_raw = raw.get("tasks")
    if not tasks_raw or not isinstance(tasks_raw, list):
        raise ValueError("Config must define a non-empty list `tasks`.")

    tasks: List[TaskSpec] = [TaskSpec(spec=t) for t in tasks_raw]

    output_root = Path(raw.get("output_root", "1_MultiRetrieval"))
    import_modules = raw.get("import_modules", []) or []
    encode_kwargs = raw.get("encode_kwargs", {}) or {}

    top_k = raw.get("top_k")
    if top_k is not None:
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer or null, got {top_k}")

    return AppConfig(
        models=models,
        tasks=tasks,
        output_root=output_root,
        import_modules=import_modules,
        encode_kwargs=encode_kwargs,
        top_k=top_k,
        use_prompts=bool(raw.get("use_prompts", False)),
        model_prompts=raw.get("model_prompts") or {},
        seed=(int(raw["seed"]) if raw.get("seed") is not None else None),
    )

# =================== TASK RESOLUTION ========================================
def safe_name(s: str) -> str:
    return s.replace("/", "__").replace(":", "_")

def import_from_path(path: str):
    """
    Import a class/function from 'module:attr' or 'module.attr'.
    """
    if ":" in path:
        mod, attr = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(f"Import path must be 'module.Class' or 'module:Class', got: {path}")
        mod, attr = ".".join(parts[:-1]), parts[-1]

    module = importlib.import_module(mod)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f"Module '{mod}' has no attribute '{attr}'") from e


def resolve_tasks(task_specs: List[TaskSpec], import_modules: List[str]) -> List[Any]:
    """
    Mix official MTEB tasks with custom classes.
    Supports task entries in the config as:
      - "OfficialTaskName"
      - "my_tasks.ChemQuest"               # import path string (no kwargs)
      - {"class": "my_tasks.ChemQuest", "kwargs": {...}}  # import path with kwargs (e.g., data_dir)
      - {"name": "OfficialTaskName"}       # explicit official by name
    """
    # Pre-imports (handy if custom tasks rely on side effects or bare names)
    for mod in import_modules:
        importlib.import_module(mod)

    resolved: List[Any] = []
    official_names: List[str] = []
    custom_items: List[Union[str, Dict[str, Any]]] = []

    for t in task_specs:
        spec = t.spec
        if isinstance(spec, str):
            if ("." in spec) or (":" in spec):
                custom_items.append(spec)
            else:
                official_names.append(spec)
        elif isinstance(spec, dict):
            # Accept {"class": "...", "kwargs": {...}} (custom) or {"name": "..."} (official)
            if "class" in spec:
                custom_items.append(spec)
            elif "name" in spec and isinstance(spec["name"], str):
                official_names.append(spec["name"])
            else:
                raise ValueError(
                    "Task dict must contain either 'class' (import path) or 'name' (official). "
                    f"Got: {spec}"
                )
        else:
            raise ValueError(f"Unsupported task spec type: {type(spec)}")

    # Resolve official tasks by name
    if official_names:
        try:
            resolved.extend(get_tasks(tasks=official_names))
        except Exception as e:
            print(f"[WARN] get_tasks failed for {official_names}: {e}")

    # Resolve custom classes
    for item in custom_items:
        if isinstance(item, str):
            cls = import_from_path(item)
            instance = cls()  # no kwargs
            resolved.append(instance)
        else:
            cls_path = item.get("class")
            if not cls_path:
                raise ValueError(f"Custom task dict missing 'class': {item}")
            kwargs = item.get("kwargs", {}) or {}
            # Expand ~ and env vars for common kwarg values like data_dir
            for k, v in list(kwargs.items()):
                if isinstance(v, str):
                    v = Path(v).expanduser()
                    kwargs[k] = str(v)
            cls = import_from_path(cls_path)
            instance = cls(**kwargs)
            resolved.append(instance)

    if not resolved:
        raise RuntimeError("No tasks resolved. Check task names/imports in config.")
    return resolved


def _task_data_files(task_specs: List["TaskSpec"]) -> List[Path]:
    """corpus / queries / qrels files of local tasks (for the hashes in the run manifest)."""
    files: List[Path] = []
    for t in task_specs:
        spec = t.spec
        if isinstance(spec, dict):
            kw = spec.get("kwargs") or {}
            d = kw.get("data_dir")
            if d:
                d = Path(str(d)).expanduser()
                names = ["corpus.jsonl", "queries.jsonl"] + ([kw["qrels_file"]] if kw.get("qrels_file") else [])
                files += [d / n for n in names]
    return files


def _local_task_labels(task_specs: List["TaskSpec"]):
    """(data_dir, qrels file name) of the first local task (Phase 2), or None."""
    for t in task_specs:
        spec = t.spec
        if isinstance(spec, dict):
            kw = spec.get("kwargs") or {}
            if kw.get("data_dir"):
                return Path(str(kw["data_dir"])).expanduser(), kw.get("qrels_file") or "qrels.jsonl"
    return None


def write_per_query_outputs(model_out_dir: Path, task_specs: List["TaskSpec"], depth: int = 100,
                            run_name: str = "run") -> Optional[Dict[str, Any]]:
    """From the saved predictions: gzipped TREC run (top ``depth``), per-query metrics, summary with the
    cross-check against the metrics MTEB stored. Returns a small dict for the manifest, or None for a task that
    has no local qrels (Phase 1 official tasks)."""
    loc = _local_task_labels(task_specs)
    preds = sorted(model_out_dir.glob("*_predictions.json"))
    if loc is None or len(preds) != 1:
        return None
    data_dir, qrels_name = loc
    run = qmetrics.load_run(preds[0])
    qrels = qmetrics.load_qrels(data_dir / qrels_name)
    qids = qmetrics.load_query_ids(data_dir / "queries.jsonl")
    n_lines = qmetrics.write_trec_run(run, model_out_dir / f"run_top{depth}.trec.gz", tag=run_name, depth=depth)
    pq = qmetrics.per_query_metrics(run, qrels, qmetrics.DEFAULT_KS, qids)
    qmetrics.write_per_query_csv(model_out_dir / "per_query_metrics.csv", pq)
    summary: Dict[str, Any] = {"qrels_file": qrels_name, "aggregate": qmetrics.aggregate(pq), "run_lines": n_lines}
    stored = sorted(model_out_dir.rglob("ChemQuest.json"))
    if stored:
        summary["cross_check"] = qmetrics.cross_check(run, qrels, stored[0], qids, max_k=depth)
        if not summary["cross_check"]["ok"]:
            print(f"[WARN] metrics cross-check against MTEB failed: {summary['cross_check']}")
    (model_out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# =================== RESULT NORMALIZATION ===================================
def normalize_results(results):
    """
    Normalize outputs from MTEB.run to a list of {task_name, main_score, raw}.
    Supports:
      - list of Pydantic TaskResult objects (newer mteb)
      - list of dicts
      - dict keyed by task name (older mteb)
    """
    out = []

    def as_dict(x):
        if hasattr(x, "model_dump"):
            try:
                return x.model_dump()
            except Exception:
                pass
        if hasattr(x, "dict"):
            try:
                return x.dict()
            except Exception:
                pass
        if isinstance(x, Mapping):
            return x
        return None

    if isinstance(results, list):
        for item in results:
            d = as_dict(item)
            if d is not None:
                task_name = (
                    d.get("task_name")
                    or d.get("task", {}).get("name")
                    or d.get("dataset", {}).get("name")
                    or "UNKNOWN_TASK"
                )
                main_score = d.get("main_score")
                out.append({"task_name": task_name, "main_score": main_score, "raw": d})
                continue

            task_name = (
                getattr(item, "task_name", None)
                or (getattr(item, "task", None).name if getattr(item, "task", None) else None)
                or "UNKNOWN_TASK"
            )
            main_score = getattr(item, "main_score", None)
            out.append({"task_name": task_name, "main_score": main_score, "raw": item})
        return out

    if isinstance(results, Mapping):
        for task_name, payload in results.items():
            d = as_dict(payload) or {}
            main_score = d.get("main_score", getattr(payload, "main_score", None))
            out.append({"task_name": task_name, "main_score": main_score, "raw": d or payload})
        return out

    out.append({"task_name": "UNKNOWN_TASK", "main_score": None, "raw": results})
    return out

# =================== MAIN ====================================================
def main():
    parser = argparse.ArgumentParser(description="Run MTEB eval from config.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to config file (.yaml/.yml/.json/.toml)",
    )
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    app_cfg = parse_app_config(raw_cfg)

    tasks_resolved = resolve_tasks(app_cfg.tasks, app_cfg.import_modules)
    task_names_for_print = []
    for t in app_cfg.tasks:
        if isinstance(t.spec, str):
            task_names_for_print.append(t.spec)
        elif isinstance(t.spec, dict):
            task_names_for_print.append(t.spec.get("class", "UNKNOWN_CUSTOM_TASK"))
        else:
            task_names_for_print.append("UNKNOWN_TASK")

    summary_rows: List[Dict[str, Any]] = []
    seeds = set_seeds(app_cfg.seed) if app_cfg.seed is not None else None

    # Ensure output root exists
    app_cfg.output_root.mkdir(parents=True, exist_ok=True)

    for cfg in app_cfg.models:
        model_name = cfg.name
        revision = cfg.revision
        normalize = cfg.normalize
        batch_size = cfg.batch_size
        # Merge global and per-model encode kwargs (model wins)
        encode_kwargs = {**app_cfg.encode_kwargs, **(cfg.encode_kwargs or {})}
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = batch_size

        print(
            f"\n=== Evaluating tasks: {', '.join(task_names_for_print)}"
            f"\nModel: {model_name} @ {revision} | batch={batch_size} | normalize={normalize}"
        )

        # Build wrapper model
        sw = Stopwatch()
        with sw.section("model_load_s"):
            base = SentenceTransformer(model_name, revision=revision,**(cfg.model_kwargs or {}))
        prompts = resolve_prompts(model_name, app_cfg.model_prompts, app_cfg.use_prompts, override=cfg.prompts)
        print(f"Prompts: {prompts if prompts else 'none'}")
        model = STNoPrompt(base, normalize=normalize, prompts=prompts)

        # Per-model output dir; MTEB will nest per-task folders inside
        model_out_dir = app_cfg.output_root / f"{safe_name(model_name)}_{revision}"
        model_out_dir.mkdir(parents=True, exist_ok=True)

        # Run all tasks for this model
        evaluation = MTEB(tasks=tasks_resolved)
        run_kwargs: Dict[str, Any] = {}
        if app_cfg.top_k is not None:
            # Only set for explicit runs (e.g. the pilot): it lowers the retrieval depth and so
            # changes the @1000 metrics. The default (None) keeps MTEB's depth.
            run_kwargs["top_k"] = app_cfg.top_k
        with sw.section("mteb_run_wall_s"):
            results = evaluation.run(
                model,
                model_name=model_name,
                model_revision=revision,
                output_folder=str(model_out_dir),
                encode_kwargs=encode_kwargs,  # replace deprecated batch_size arg
                # Save the ranked retrieval results (<task>_default_predictions.json in output_folder).
                save_predictions=True,
                **run_kwargs,
            )

        # MTEB silently reuses an existing ChemQuest.json and then saves nothing: fail loudly instead.
        if not list(model_out_dir.glob("*_predictions.json")):
            raise RuntimeError(
                f"No predictions were saved in {model_out_dir}. MTEB skips tasks whose results already "
                "exist: point stage6.output_root to a new folder to re-run."
            )

        per_query = write_per_query_outputs(model_out_dir, app_cfg.tasks, depth=app_cfg.top_k or 100)

        # ---- run manifest (config, git, packages, hardware, prompts, data hashes, timing); written before summary.json
        mteb_eval_s = [getattr(r, "evaluation_time", None) for r in (results if isinstance(results, list) else [])]
        mteb_eval_s = [x for x in mteb_eval_s if isinstance(x, (int, float))]
        timing = {
            **sw.seconds,
            # MTEB's own evaluation_time: encode corpus + queries, search, metrics, predictions dump (not model load)
            "mteb_evaluation_time_s": sum(mteb_eval_s) if mteb_eval_s else None,
            **summarize_encode_timings(getattr(model, "timings", [])),
        }
        if timing["mteb_evaluation_time_s"] is not None:
            timing["search_metrics_and_dump_s"] = timing["mteb_evaluation_time_s"] - timing["encode_passage_s"] - timing["encode_query_s"]
        write_manifest(model_out_dir / "run_manifest.json", build_manifest(
            repo_root=Path(__file__).resolve().parents[1],
            config_path=args.config,
            resolved_config=raw_cfg,
            model={"name": model_name, "revision": revision, "normalize": normalize, "model_kwargs": cfg.model_kwargs},
            use_prompts=app_cfg.use_prompts,
            prompts=prompts,
            batch_size=batch_size,
            seeds=seeds,
            data_files=_task_data_files(app_cfg.tasks),
            timing=timing,
            extra={"top_k": app_cfg.top_k, "per_query_outputs": per_query, "mteb_version": getattr(sys.modules.get("mteb"), "__version__", None)},
        ))

        per_task = normalize_results(results)
        for r in per_task:
            tname = r["task_name"]
            score = r["main_score"]
            summary_rows.append(
                {
                    "model": model_name,
                    "revision": revision,
                    "normalize": normalize,
                    "batch_size": batch_size,
                    "task": tname,
                    "main_score": score,
                }
            )
            print(f"  • {tname}: main_score={score}")

    # Combined summaries
    summary_json = app_cfg.output_root / "summary.json"
    summary_csv = app_cfg.output_root / "summary.csv"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    headers = ["model", "revision", "normalize", "batch_size", "task", "main_score"]
    with summary_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in summary_rows:
            f.write(",".join([str(row.get(h, "")) if row.get(h, "") is not None else "" for h in headers]) + "\n")

    print("\n=== Summary ===")
    for row in summary_rows:
        print(
            f"{row['model']} @ {row['revision']} | batch={row['batch_size']} | "
            f"norm={row['normalize']} | {row['task']} main_score={row['main_score']}"
        )

if __name__ == "__main__":
    main()
