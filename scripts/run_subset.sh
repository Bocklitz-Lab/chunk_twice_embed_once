#!/usr/bin/env bash
# Subset of the full rerun (a CHECK, not a result): 4 dense models x 35 configurations + BM25 / hybrid / rerank, full Phase 1 on the second GPU,
# a 1-configuration transfer run, report, analysis, figures, annotation sheets and the comparison tables. Unattended; every cut rule is in this file.
#
#   nohup scripts/run_subset.sh > artifacts/subset_v1/driver.log 2>&1 &
#   cat results/v2-revision/subset/STATUS.md            # progress; "STATE: done" at the end
#   SUBSET_DRY=1 scripts/run_subset.sh                  # dry run: 1 model, 1 configuration, no Phase 1 / transfer (a few minutes)
#
# Clock (hours from the start of this script): prepare 0-0.75, dense 0.75-4 (GPU 0), Phase 1 0-4 (GPU 1), rerank + stage 2 + transfer 4-6.5,
# report / analysis / figures 6.5-8, comparison 8-8.5. CUT RULES (applied here, not by hand):
#   hour 4    a partial summary is written into STATUS.md
#   hour 5    dense runs of e5-small that are not finished are dropped (e5-small is never started after hour 5)
#   hour 6.5  unfinished rerank and transfer runs are stopped; the report goes on with what exists
#   hour 9    remaining analysis steps are stopped, subset_compare.py runs on what exists, STATUS.md = done
# Never dropped: a chunking strategy, a size of the size sweep.
set -u
cd "$(dirname "$0")/.."
export PATH="$PWD/myenv/bin:$PATH"
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
START=$(date +%s)
DRY=${SUBSET_DRY:-0}

if [ "$DRY" = 1 ]; then
  ROOT=artifacts/subset_dry; OUT=results/v2-revision/subset_dry; TR=artifacts/subset_dry_transfer
  MODELS=(intfloat/e5-small); STRATS=(fixed_token); SIZES_A="128"; RES=200
  H_E5SMALL=99; H_RERANK=99; H_ANALYSIS=99
else
  ROOT=artifacts/subset_v1; OUT=results/v2-revision/subset; TR=artifacts/subset_v1_transfer
  # largest first (bge-m3 568M, e5-large-v2 335M, nomic-v1.5 137M, e5-small 33M)
  MODELS=(BAAI/bge-m3 intfloat/e5-large-v2 nomic-ai/nomic-embed-text-v1.5 intfloat/e5-small)
  STRATS=(recursive_token fixed_token semantic_recursive semantic_fixed hierarchical_section); SIZES_A="128 256 384 448 512"; RES=10000
  H_E5SMALL=5; H_RERANK=6.5; H_ANALYSIS=9
fi
CFG=$ROOT/configs/main_subset.yaml; CFGB=$ROOT/configs/main_subset_b.yaml; STG=$ROOT/configs/stages
LOGS=$ROOT/logs; STATUS=$OUT/STATUS.md; STEPS=$OUT/steps.csv; LOCK=$ROOT/.status.lock
mkdir -p "$ROOT" "$OUT" "$LOGS" "$ROOT/configs" "$OUT/analysis" "$OUT/validation" "$OUT/figures"
[ -f "$STEPS" ] || echo "step,start,end,seconds,status,runs,note" > "$STEPS"

hours() { awk -v n="$(date +%s)" -v s="$START" 'BEGIN{printf "%.2f", (n-s)/3600}'; }
left_s() { awk -v n="$(date +%s)" -v s="$START" -v h="$1" 'BEGIN{x=h*3600-(n-s); if (x<60) x=60; printf "%d", x}'; }   # seconds until hour $1 (at least 60)
after() { awk -v a="$(hours)" -v h="$1" 'BEGIN{exit !(a>=h)}'; }                                                     # true if the clock is past hour $1
slug() { echo "$1" | sed 's/[^0-9A-Za-z]/_/g'; }
n_runs() { find "$ROOT/tests" -path "*/results/summary.json" -path "*$(slug "$1")_*" 2>/dev/null | wc -l; }

write_status() {   # STATUS.md from steps.csv (called under the lock)
  python - "$STEPS" "$STATUS" "$START" "$1" <<'PY'
import csv, sys, time
steps, status, start, state = sys.argv[1:5]
rows = list(csv.DictReader(open(steps)))
el = (time.time() - int(start)) / 3600
out = [f"# Subset run status (a CHECK on a SUBSET: 4 models x 35 configurations; nothing here is a final number)", "",
       f"STATE: {state}", f"elapsed: {el:.2f} h since the start of the driver", "", "| step | start | end | seconds | status | runs | note |", "|---|---|---|---|---|---|---|"]
out += ["| " + " | ".join(r[k] for k in ("step", "start", "end", "seconds", "status", "runs", "note")) + " |" for r in rows]
open(status, "w").write("\n".join(out) + "\n")
PY
}

# step NAME TIMEOUT_SECONDS NOTE -- command ...   (logs to $LOGS/NAME.log; a failure puts the last 30 lines into STATUS.md and the driver goes on)
step() {
  local name="$1" to="$2" note="${3//,/;}"; shift 4
  local log="$LOGS/$(echo "$name" | tr ':/ ' '___').log" t0 t1 rc st
  t0=$(date +%s)
  timeout -k 30 "$to" "$@" > "$log" 2>&1; rc=$?
  t1=$(date +%s)
  st=ok; [ $rc -eq 124 ] && st="stopped by timeout"; [ $rc -ne 0 ] && [ $rc -ne 124 ] && st="FAILED rc=$rc"
  local runs=""; case "$name" in dense:*) runs=$(n_runs "$(echo "$name" | cut -d: -f2)");; esac
  {
    flock -x 9
    echo "$name,$(date -d @$t0 +%H:%M:%S),$(date -d @$t1 +%H:%M:%S),$((t1-t0)),$st,$runs,$note (clock h $(hours))" >> "$STEPS"
    if [ "$st" != ok ]; then { echo; echo "### $name: $st (log: $log)"; echo '```'; tail -n 30 "$log" | cut -c1-300; echo '```'; } >> "$OUT/failures.md"; fi
    write_status running
  } 9> "$LOCK"
  return 0
}

notice() { for d in "$@"; do mkdir -p "$d"; echo "SUBSET CHECK: 4 dense models x 35 configurations (+ BM25, hybrid, rerank); a check before the full rerun, not a result for the paper." > "$d/SUBSET_NOTICE.txt"; done; }
notice "$ROOT" "$OUT" "$OUT/analysis" "$OUT/figures" "$OUT/validation" "$ROOT/report"
: > "$OUT/failures.md"; write_status running

# ---------------------------------------------------------------------------------------------------- configuration (derived from the main config; nothing in configs/ is changed)
python - "$ROOT" "$OUT" "$TR" "$CFG" "$CFGB" "${MODELS[*]}" "$DRY" <<'PY'
import sys, yaml, copy
root, out, tr, cfg, cfgb, models, dry = sys.argv[1:8]
doc = yaml.safe_load(open("configs/main_config.yaml"))
p = doc["params"]
p["paths"]["artifacts_root"] = root
p["transfer"]["artifacts_root"] = tr
p["transfer"]["report_dir"] = f"{out}/transfer"
p["transfer"]["configurations"] = [c for c in p["transfer"]["configurations"] if c["id"] == "C1"]
p["report"]["out_dir"] = out
p["analysis"]["out_dir"] = f"{out}/analysis"
p["analysis"]["enabled_in_report"] = False
p["efficiency"]["latency_cells"] = [c for c in p["efficiency"]["latency_cells"] if c["chunker"] == "fixed_token" and c["size"] in (256, 512)] or p["efficiency"]["latency_cells"][:1]
if dry == "1":
    p["efficiency"]["latency_cells"] = [{"chunker": "fixed_token", "size": 128, "overlap": 0}]
    p["rerank"]["models"] = models.split()[:1]
    p["rerank"]["ablation"] = []
yaml.safe_dump(doc, open(cfg, "w"), sort_keys=False, width=1000)
doc2 = copy.deepcopy(doc)
doc2["params"]["rerank"]["ablation"] = []          # the ablation cell (fixed-token 256/0) is in group A only
yaml.safe_dump(doc2, open(cfgb, "w"), sort_keys=False, width=1000)
PY
python -m pipeline_lib.split_config --input "$CFG" --output-dir "$STG" --sections stage1 stage2 stage3 > "$LOGS/split_config.log" 2>&1
# model@revision strings from the stage 1 pins
REVS=$(python - "$STG/stage1.yaml" "${MODELS[*]}" <<'PY'
import sys, yaml
pins = {m["name"]: m["revision"] for m in yaml.safe_load(open(sys.argv[1]))["models"]}
print(" ".join(f"{n}@{pins[n]}" for n in sys.argv[2].split()))
PY
)
python - <<PY | tee "$OUT/subset_cells.txt"
from pipeline_lib import grid
A = [(int(x), 0) for x in "$SIZES_A".split()]
B = [(256, 32), (256, 64)] if "$DRY" != "1" else []
bad = [c for c in A + B if not grid.is_valid(*c)]
print("sizes/overlaps of the subset:", A + B, "| invalid pairs:", bad)
assert not bad, bad
PY
grep -q "invalid pairs: \[\]" "$OUT/subset_cells.txt" || { echo "invalid cell"; exit 1; }
if [ "$DRY" = 1 ]; then
  H0=$(python - <<'PY'
import csv, glob
r = []
for f in glob.glob("results/v2-revision/pilot/runs_overview.csv"):
    for x in csv.DictReader(open(f)):
        if x.get("mteb_evaluation_time_s"): r.append(float(x["mteb_evaluation_time_s"]))
print(f"pilot: mean dense/BM25 evaluation_time {sum(r)/max(1,len(r)):.0f} s over {len(r)} runs" if r else "no pilot timings")
PY
)
else
  # estimate from the pilot (dense ~100 s per run for a large model): printed, and written to STATUS
  H0=$(python - <<'PY'
import csv
rows = [x for x in csv.DictReader(open("results/v2-revision/pilot/runs_overview.csv")) if x.get("mteb_evaluation_time_s") and not x["model"].startswith(("bm25", "hybrid"))]
t = {}
for x in rows: t.setdefault(x["model"], []).append(float(x["mteb_evaluation_time_s"]))
m = sum(sum(v) / len(v) for v in t.values()) / len(t)
print(f"estimate from the pilot: dense run {m:.0f} s (large model, includes load/probes ~ +30 %) -> 4 models x 35 runs ~ {m*1.3*35*2.2/3600:.1f} h; Phase 1 ~ 3 h on GPU 1")
PY
)
fi
echo "$H0"; echo "ESTIMATE: $H0" >> "$OUT/failures.md"

CH_ARGS=(--chunkers "${STRATS[@]}")
RERUN="python -m stages.rerun"
common=(--main-config "$CFG" --artifacts-root "$ROOT")
mods=(--models ${REVS})
FLAG_OFF=(--no-report)

# ---------------------------------------------------------------------------------------------------- Phase 1 on GPU 1 (background chain), then stage 2, external BM25, transfer
gpu1_chain() {
  step "stage3_download" 1800 "ChemQuests pinned release + question_metadata" -- python -m stages.download_chemquests --config "$STG/stage3.yaml"
  [ "$DRY" = 1 ] && return
  step "phase1" 18000 "41 models, both tasks, split from params.phase1.eval_split, prompts on" -- env CUDA_VISIBLE_DEVICES=1 python -m stages.model_screening --config "$STG/stage1.yaml" --efficiency
  step "stage2" 600 "shortlist" -- python -m stages.screen_top_k_models --config "$STG/stage2.yaml"
  step "external_bm25" 3600 "BM25 on the two external tasks" -- python -m stages.run_bm25_mteb --main-config "$CFG"
  while [ ! -f "$ROOT/.prepared" ]; do sleep 20; if after 6; then break; fi; done
  step "transfer" "$(left_s $H_RERANK)" "1 configuration (fixed-token 512/0), all transfer models" -- env CUDA_VISIBLE_DEVICES=1 python -m stages.transfer --main-config "$CFG" --efficiency --latency
}

# ---------------------------------------------------------------------------------------------------- GPU 0 chain: prepare, sheets, BM25, dense, hybrid, rerank
gpu0_chain() {
  step "stage3_wait" 3600 "wait for the data download" -- bash -c "until [ -f $ROOT/chemquests_hf_db/question_metadata.jsonl ]; do sleep 10; done"
  step "prepare:A" 7200 "5 strategies x 5 sizes at overlap 0" -- $RERUN prepare "${common[@]}" "${mods[@]}" "${CH_ARGS[@]}" --sizes $SIZES_A --overlaps 0 "${FLAG_OFF[@]}"
  [ "$DRY" = 1 ] || step "prepare:B" 3600 "256 / 32 and 256 / 64" -- $RERUN prepare "${common[@]}" "${mods[@]}" "${CH_ARGS[@]}" --sizes 256 --overlaps 32 64 "${FLAG_OFF[@]}"
  touch "$ROOT/.prepared"
  if [ "$DRY" != 1 ]; then
    step "annotation_sheets" 600 "validation_sample + question_features_check (hour 1)" -- bash -c "python -m tools.validation_sample --main-config $CFG --qa $ROOT/chemquests_hf_db/qa.jsonl --prepared-root $ROOT/prepared --out-dir $OUT/validation && python -m tools.question_features_check draw --main-config $CFG --qa $ROOT/chemquests_hf_db/qa.jsonl --out-dir $OUT/validation"
  fi
  step "bm25:A" 3600 "BM25 baseline, CPU" -- $RERUN eval "${common[@]}" --models bm25@rank_bm25 "${CH_ARGS[@]}" --sizes $SIZES_A --overlaps 0 --no-hybrid --no-rerank "${FLAG_OFF[@]}"
  [ "$DRY" = 1 ] || step "bm25:B" 1800 "BM25 baseline, CPU" -- $RERUN eval "${common[@]}" --models bm25@rank_bm25 "${CH_ARGS[@]}" --sizes 256 --overlaps 32 64 --no-hybrid --no-rerank "${FLAG_OFF[@]}"
  for spec in ${REVS}; do
    name=${spec%@*}
    if [ "$name" = "intfloat/e5-small" ] && after $H_E5SMALL; then echo "$name,-,-,0,skipped (hour $H_E5SMALL cut rule),0,dense not started" >> "$STEPS"; continue; fi
    to=14400; [ "$name" = "intfloat/e5-small" ] && to=$(left_s $H_E5SMALL)
    step "dense:$name:A" $to "dense, 25 runs, prompts on, efficiency + latency (2 cells)" -- env CUDA_VISIBLE_DEVICES=0 $RERUN eval "${common[@]}" --models "$spec" --no-bm25 "${CH_ARGS[@]}" --sizes $SIZES_A --overlaps 0 --no-hybrid --no-rerank "${FLAG_OFF[@]}" --efficiency --latency
    [ "$DRY" = 1 ] || step "dense:$name:B" $to "dense, 10 runs" -- env CUDA_VISIBLE_DEVICES=0 $RERUN eval "${common[@]}" --models "$spec" --no-bm25 "${CH_ARGS[@]}" --sizes 256 --overlaps 32 64 --no-hybrid --no-rerank "${FLAG_OFF[@]}" --efficiency --latency
  done
  step "hybrid:A" 3600 "RRF of every dense run with BM25" -- $RERUN hybrid "${common[@]}" "${mods[@]}" bm25@rank_bm25 "${CH_ARGS[@]}" --sizes $SIZES_A --overlaps 0 "${FLAG_OFF[@]}"
  [ "$DRY" = 1 ] || step "hybrid:B" 1800 "RRF" -- $RERUN hybrid "${common[@]}" "${mods[@]}" bm25@rank_bm25 "${CH_ARGS[@]}" --sizes 256 --overlaps 32 64 "${FLAG_OFF[@]}"
  step "rerank:A" "$(left_s $H_RERANK)" "e5-large-v2 depth 50 + ablation 20 / 100 at fixed-token 256/0" -- env CUDA_VISIBLE_DEVICES=0 $RERUN rerank "${common[@]}" "${mods[@]}" "${CH_ARGS[@]}" --sizes $SIZES_A --overlaps 0 "${FLAG_OFF[@]}"
  [ "$DRY" = 1 ] || step "rerank:B" "$(left_s $H_RERANK)" "e5-large-v2 depth 50" -- env CUDA_VISIBLE_DEVICES=0 $RERUN rerank --main-config "$CFGB" --artifacts-root "$ROOT" "${mods[@]}" "${CH_ARGS[@]}" --sizes 256 --overlaps 32 64 "${FLAG_OFF[@]}"
}

# ---------------------------------------------------------------------------------------------------- hour-4 partial summary
partial_at_4h() {
  [ "$DRY" = 1 ] && return
  sleep "$(left_s 4)"
  python - "$ROOT" "$OUT" <<'PY' > "$OUT/partial_hour4.md" 2>&1
import csv, glob, os, sys, json
root, out = sys.argv[1:3]
runs = glob.glob(f"{root}/tests/*/results/summary.json")
kinds = {"bm25": 0, "hybrid": 0, "rerank": 0, "dense": 0}
rows = []
for f in runs:
    d = os.path.dirname(os.path.dirname(f)); name = os.path.basename(d)
    k = "bm25" if name.startswith("bm25") else "hybrid" if name.startswith("hybrid") else "rerank" if name.startswith("rerank") else "dense"
    kinds[k] += 1
    pq = glob.glob(f"{d}/results/*/per_query_metrics.csv")
    if k == "dense" and pq:
        v = list(csv.DictReader(open(pq[0])))
        nd = sum(float(x["ndcg_at_10"]) for x in v) / len(v); rc = sum(float(x["recall_at_10"]) for x in v) / len(v)
        rows.append((name, (nd * rc) ** 0.5))
print("# Partial summary at hour 4 (subset check)\n")
print("finished runs:", kinds)
print("\nfirst numbers: mean Geom@10 (span labels) per model, finished dense runs\n")
by = {}
for n, g in rows:
    m = n.split("_fixed_token")[0].split("_recursive_token")[0].split("_semantic")[0].split("_hierarchical")[0]
    by.setdefault(m, []).append(g)
for m, v in sorted(by.items()):
    print(f"- {m}: {sum(v)/len(v):.3f} over {len(v)} runs")
print("\nPhase 1:", "done" if os.path.exists(f"{root}/embedding_models_screening/passed_per_dataset__validation_prompts.csv") else "not finished",
      "| failures so far:", open(f"{out}/failures.md").read().count("###"))
PY
  { flock -x 9; { echo; echo "## Partial summary at hour 4"; cat "$OUT/partial_hour4.md"; } >> "$OUT/failures.md"; } 9> "$LOCK"
}

partial_at_4h &
gpu1_chain > "$LOGS/chain_gpu1.log" 2>&1 &
P1=$!
gpu0_chain > "$LOGS/chain_gpu0.log" 2>&1
wait $P1

# ---------------------------------------------------------------------------------------------------- report, analysis, figures (6.5 - 8 h)
ext=(--external-root "$ROOT/embedding_models_screening/validation_prompts" --external-root "$ROOT/embedding_models_screening/baselines")
step "make report" 3600 "paper-level bootstrap, $RES resamples" -- python -m tools.make_report --artifacts-root "$ROOT" --out-dir "$OUT" --main-config "$CFG" --resamples $RES --no-analysis "${ext[@]}"
if tail -n 3 "$STEPS" | grep -q "make report.*\(timeout\|FAILED\)"; then
  RES=2000
  step "make report" 3600 "RETRY with 2,000 resamples (the 10,000 run took too long)" -- python -m tools.make_report --artifacts-root "$ROOT" --out-dir "$OUT" --main-config "$CFG" --resamples $RES --no-analysis "${ext[@]}"
fi
step "make analysis" "$(left_s $H_ANALYSIS)" "all analysis tools on the main root, $RES resamples" -- python -m tools.run_analysis --main-config "$CFG" --roots "$ROOT" --out-dir "$OUT/analysis" --resamples $RES
step "figures" 1800 "figures with baselines, bands, best model" -- python -m tools.figures --runs "$ROOT/report/runs_overview.csv" --out "$OUT/figures" --roots "$ROOT" --qa "$ROOT/chemquests_hf_db/qa.jsonl" --include-baselines --best-model --main-config "$CFG" --resamples $RES
notice "$ROOT/report" "$OUT" "$OUT/analysis" "$OUT/figures" "$OUT/validation" "$OUT/transfer"
step "subset_compare" 3600 "scripts/subset_compare.py" -- python scripts/subset_compare.py --root "$ROOT" --out "$OUT" --resamples $RES
{ flock -x 9; write_status "done ($(hours) h after the start)"; } 9> "$LOCK"
echo "done"
