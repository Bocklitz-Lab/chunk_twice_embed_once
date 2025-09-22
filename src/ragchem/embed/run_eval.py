
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

# 3rd-party
from sentence_transformers import SentenceTransformer
from mteb import MTEB, get_tasks

# local
from pipeline_lib.model_wrapper import STNoPrompt  # keep your wrapper

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

    return AppConfig(
        models=models,
        tasks=tasks,
        output_root=output_root,
        import_modules=import_modules,
        encode_kwargs=encode_kwargs,
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
        base = SentenceTransformer(model_name, revision=revision,**(cfg.model_kwargs or {}))
        model = STNoPrompt(base, normalize=normalize)

        # Per-model output dir; MTEB will nest per-task folders inside
        model_out_dir = app_cfg.output_root / f"{safe_name(model_name)}_{revision}"
        model_out_dir.mkdir(parents=True, exist_ok=True)

        # Run all tasks for this model
        evaluation = MTEB(tasks=tasks_resolved)
        results = evaluation.run(
            model,
            model_name=model_name,
            model_revision=revision,
            output_folder=str(model_out_dir),
            encode_kwargs=encode_kwargs,  # replace deprecated batch_size arg
        )

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
