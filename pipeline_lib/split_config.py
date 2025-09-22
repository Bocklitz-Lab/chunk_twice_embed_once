#!/usr/bin/env python3
"""
Split a sectioned YAML into per-stage YAMLs, with param-first overrides and full resolution.

What it does
------------
1) Load combined YAML.
2) Apply --param KEY=VALUE to params.KEY (VALUE is YAML-parsed).
3) Resolve Stage 4 strategy from params.strategy_presets (and optional overrides).
4) Interpolate any ${...} placeholders across the *entire* document using the updated params,
   repeating until fully resolved (fixed-point interpolation).
5) Write per-stage YAMLs with concrete values (no ${...} left).
   While dumping, only escape control characters so strings like '\\n' remain literal where intended
   without over-quoting everything.
"""

import argparse
import copy
import os
import re
import sys
from typing import Any, Dict, Iterable, Tuple

try:
    import yaml
except Exception:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    raise

PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


# ---------------------------
# YAML Dumper: minimal quoting, escape control chars only
# ---------------------------
class EscapingDumper(yaml.SafeDumper):
    """Custom dumper that escapes control characters in strings,
    but otherwise keeps YAML minimal and human-friendly."""
    pass


def _represent_str_escaped(dumper: yaml.SafeDumper, data: str):
    """
    - If string contains *actual* control characters (\n, \r, \t), replace them with two-char
      sequences '\\n', '\\r', '\\t' so they survive round-trips literally.
    - Prefer single-quoted style for such strings (no backslash processing), fall back to double
      quotes if the string itself contains a single quote.
    - Otherwise, use the default string representer (no forced quoting).
    """
    has_control = any(c in data for c in ("\n", "\r", "\t"))
    if has_control:
        escaped = data.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        style = "'" if "'" not in escaped else '"'
        return dumper.represent_scalar('tag:yaml.org,2002:str', escaped, style=style)

    # Default behavior: minimal quoting
    return yaml.representer.SafeRepresenter.represent_str(dumper, data)


EscapingDumper.add_representer(str, _represent_str_escaped)


# ---------------------------
# Helpers
# ---------------------------
def parse_param(arg: str) -> Tuple[str, Any]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(f"--param must be KEY=VALUE, got: {arg}")
    k, v = arg.split("=", 1)
    try:
        v_parsed = yaml.safe_load(v)
    except Exception:
        v_parsed = v
    return k.strip(), v_parsed


def deep_get(root: Any, dotpath: str) -> Any:
    cur = root
    for seg in [s for s in dotpath.split(".") if s]:
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            raise KeyError(dotpath)
    return cur


def deep_set(root: Dict[str, Any], dotpath: str, value: Any) -> None:
    cur = root
    parts = [s for s in dotpath.split(".") if s]
    if not parts:
        raise ValueError("Empty dotpath")
    for seg in parts[:-1]:
        if not isinstance(cur, dict):
            raise ValueError(f"Cannot set at '{seg}': parent is not a mapping")
        cur = cur.setdefault(seg, {})
    if not isinstance(cur, dict):
        raise ValueError("Cannot set: parent is not a mapping")
    cur[parts[-1]] = value


def deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        out = dict(base)
        for k, v in patch.items():
            out[k] = deep_merge(out.get(k), v) if k in out else copy.deepcopy(v)
        return out
    return copy.deepcopy(patch)


# ---------------------------
# Interpolation (fixed-point)
# ---------------------------
def _interpolate_once(val: Any, ctx: Dict[str, Any]) -> Any:
    """Single-pass substitution of ${...} placeholders."""
    if isinstance(val, str):
        def _sub(m):
            key = m.group(1).strip()
            if "|" in key:
                path, default_literal = key.split("|", 1)
                path = path.strip()
                default_literal = default_literal.strip()
                try:
                    got = deep_get(ctx, path)
                    return str(got)
                except KeyError:
                    return default_literal
            try:
                got = deep_get(ctx, key)
                return str(got)
            except KeyError:
                raise KeyError(f"Unresolved placeholder: ${{{key}}}")
        return PLACEHOLDER_RE.sub(_sub, val)

    if isinstance(val, list):
        return [_interpolate_once(x, ctx) for x in val]
    if isinstance(val, dict):
        return {k: _interpolate_once(v, ctx) for k, v in val.items()}
    return val


def _has_placeholders(x: Any) -> bool:
    if isinstance(x, str):
        return bool(PLACEHOLDER_RE.search(x))
    if isinstance(x, list):
        return any(_has_placeholders(i) for i in x)
    if isinstance(x, dict):
        return any(_has_placeholders(v) for v in x.values())
    return False


def interpolate(val: Any, ctx: Dict[str, Any], max_passes: int = 10) -> Any:
    """
    Repeat interpolation until no ${...} remain or max_passes reached.
    Also re-resolves the context each pass so chained references settle globally.
    """
    out = val
    local_ctx = ctx
    for _ in range(max_passes):
        out = _interpolate_once(out, local_ctx)
        if isinstance(local_ctx, dict):
            local_ctx = _interpolate_once(local_ctx, local_ctx)
        if not _has_placeholders(out):
            break
    return out


# ---------------------------
# Stage 4 strategy resolution
# ---------------------------
def resolve_stage4_strategy(doc: Dict[str, Any]) -> None:
    """Replace stage4.strategy with chosen preset + overrides from params or stage4."""
    params = doc.get("params", {})
    presets = params.get("strategy_presets", {}) or {}
    s4 = doc.get("stage4")
    if not isinstance(s4, dict):
        return

    selected_name = None
    overrides: Dict[str, Any] = {}

    if "strategy" in s4:
        strat = s4["strategy"]
        if isinstance(strat, str):
            selected_name = strat
        elif isinstance(strat, dict):
            selected_name = strat.get("preset", selected_name)
            overrides = strat.get("overrides") or {}
    if not selected_name:
        pstrat = params.get("strategy")
        if isinstance(pstrat, dict):
            selected_name = pstrat.get("preset", selected_name)
            overrides = deep_merge(overrides, pstrat.get("overrides") or {})

    if not selected_name:
        return  # nothing to do

    preset = presets.get(selected_name)
    if not preset:
        raise KeyError(f"Stage4 strategy preset '{selected_name}' not found in params.strategy_presets")

    resolved = deep_merge(preset, overrides)
    s4["strategy"] = resolved


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main(argv: Iterable[str] = None) -> int:
    ap = argparse.ArgumentParser(description="Split config with param-first overrides and concrete stage files.")
    ap.add_argument("-i", "--input", required=True, help="Combined YAML")
    ap.add_argument("-o", "--output-dir", required=True, help="Output folder for per-stage YAMLs")
    ap.add_argument("-s", "--sections", nargs="*", default=None, help="Only export these sections (default: all)")
    ap.add_argument("--filename-template", default="{section}.yaml", help="Template for filenames")
    ap.add_argument("--param", dest="params", action="append", default=[], type=parse_param,
                    metavar="KEY=VALUE",
                    help="Override params.KEY=VALUE (VALUE YAML-parsed). Ex: --param model.name='intfloat/e5-base-v2'")
    args = ap.parse_args(argv)

    # Load
    with open(args.input, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        print("ERROR: top-level YAML must be a mapping with sections.", file=sys.stderr)
        return 2

    # Ensure params exists
    if "params" not in doc or not isinstance(doc["params"], dict):
        doc["params"] = {}

    # 1) Apply user overrides into params.*
    for k, v in args.params:
        deep_set(doc["params"], k, v)

    # 2) First full interpolation (fixed-point) across the entire doc
    doc = interpolate(doc, ctx=doc)

    # 3) Resolve Stage 4 strategy preset (preset + overrides => final concrete strategy object)
    resolve_stage4_strategy(doc)

    # 4) Interpolate again (fixed-point) in case overrides introduced new ${...} strings
    doc = interpolate(doc, ctx=doc)

    # 5) Write selected sections with concrete values
    wanted = args.sections or [k for k in doc.keys() if k != "params"]
    ensure_dir(args.output_dir)

    for sec in wanted:
        if sec not in doc:
            print(f"ERROR: section '{sec}' not found.", file=sys.stderr)
            return 2
        out_name = args.filename_template.format(section=sec)
        out_path = os.path.join(args.output_dir, out_name)
        ensure_dir(os.path.dirname(out_path))

        # Final guarantee: section is fully interpolated before dumping
        section_data = interpolate(doc[sec], ctx=doc)

        with open(out_path, "w", encoding="utf-8") as wf:
            yaml.dump(
                section_data,
                wf,
                sort_keys=False,
                allow_unicode=True,
                Dumper=EscapingDumper,
                default_flow_style=False,  # block style (no inline JSON-y flow)
                width=4096                 # avoid line folding surprises
            )
        print(f"Wrote {sec}: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
