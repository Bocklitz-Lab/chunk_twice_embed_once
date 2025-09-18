#!/usr/bin/env python3
import csv, json, os
from pipeline_lib.utils import load_config, cfg_get, ensure_dir, save_json

def main():
    cfg = load_config("config.yaml")
    qa_csv = cfg_get(cfg, "stage2_qa.qa_csv", required=True)
    out_dir = cfg_get(cfg, "stage2_qa.output_dir", required=True)
    map_path = os.path.join(cfg_get(cfg, "stage1_corpus.output_dir", required=True), "corpus_id_mapping.json")
    ensure_dir(out_dir)

    with open(map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    qa_pairs = []
    with open(qa_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                refs = json.loads(row["references"])
            except json.JSONDecodeError:
                print(f"Bad references JSON: {row.get('references')}")
                continue
            corpus_id = row["corpus_id"].strip()
            if corpus_id in id_map:
                qa_pairs.append([row["question"].strip(), refs, id_map[corpus_id]])
            else:
                print(f"Skipping unmatched corpus_id {corpus_id}")

    save_json(qa_pairs, os.path.join(out_dir, "qa_pairs.json"))
    print("Stage2 done.")

if __name__ == "__main__":
    main()
