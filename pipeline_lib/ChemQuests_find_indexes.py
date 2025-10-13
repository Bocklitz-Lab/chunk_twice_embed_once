# SPDX-License-Identifier: CC-BY-4.0

import pandas as pd
import json
import os
import re
import unicodedata
from rapidfuzz import fuzz

# ============ 🔧 TUNABLE PARAMETERS ============
BUFFER = 10  # Extra characters before/after the extracted snippet
SHIFT_RANGE = 10  # Range of character shifts to check for better alignment
SIMILARITY_THRESHOLD = 80  # Min similarity to consider as a match

# ============ 📂 FILE PATHS ============
TEXT_FOLDER = "./../full-text"
INPUT_CSV = "chunking-eval-query-snippet-cleaned.csv"
OUTPUT_CSV = "updated_chunking_eval.csv"  # Updated dataset after removing mismatches
LOG_FILE = "verification_report.txt"

# Load the dataset
df = pd.read_csv(INPUT_CSV)

# Initialize log
mismatched_indices = []  # Store rows to delete
correct_matches = {}  # Store index and corrected snippets

with open(LOG_FILE, "w", encoding="utf-8") as log:
    log.write("🔍 Verification Report\n")
    log.write("=" * 50 + "\n\n")

# ============ 🔄 TEXT NORMALIZATION FUNCTION ============
def normalize_text(text):
    """Normalize text by lowercasing, removing extra spaces, normalizing unicode, and cleaning symbols."""
    if text is None:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)  # Normalize special characters
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    text = re.sub(r"[\$\{\}\^]", "", text)  # Remove LaTeX-like symbols
    text = text.replace("−", "-")  # Replace special minus sign with normal "-"
    return text

# ============ 🔍 VERIFICATION PROCESS ============
for index, row in df.iterrows():
    doc_id = str(row["corpus_id"]).split("/")[-1].replace(".txt", "")  # Extract doc_id from path
    text_file = os.path.join(TEXT_FOLDER, f"{doc_id}.txt")

    # Load the corresponding text document
    if not os.path.exists(text_file):
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"❌ ERROR: {text_file} not found (Deleting entry {index})\n")
        mismatched_indices.append(index)
        continue

    with open(text_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Extract start and end index from references column
    try:
        references = json.loads(row["references"])[0]  # Extract first reference
        start_idx = references.get("start_index")
        end_idx = references.get("end_index")
        stored_snippet = references["content"]
    except (json.JSONDecodeError, IndexError, KeyError):
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"❌ ERROR: Malformed references field in entry {index} (Deleting entry)\n")
        mismatched_indices.append(index)
        continue

    # Ensure indices are valid
    if start_idx is None or end_idx is None or start_idx >= end_idx or end_idx > len(full_text):
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"❌ ERROR: Invalid indices for doc_id={doc_id}, entry {index} (Deleting entry)\n")
        mismatched_indices.append(index)
        continue

    # Extract multiple shifted versions of the snippet
    best_similarity = 0
    best_extracted_snippet = ""

    for shift in range(-SHIFT_RANGE, SHIFT_RANGE + 1):
        shifted_start = max(0, start_idx + shift - BUFFER)
        shifted_end = min(len(full_text), end_idx + shift + BUFFER)
        extracted_snippet = normalize_text(full_text[shifted_start:shifted_end])
        stored_snippet = normalize_text(stored_snippet)

        # Compute similarity
        similarity = fuzz.ratio(extracted_snippet, stored_snippet)

        # Keep track of the best similarity found
        if similarity > best_similarity:
            best_similarity = similarity
            best_extracted_snippet = extracted_snippet

        # ✅ If stored snippet is a substring of the extracted text, accept it
        if stored_snippet in extracted_snippet:
            best_similarity = 100  # Force acceptance
            break

    # ✅ Accept match if:
    if best_similarity >= SIMILARITY_THRESHOLD or stored_snippet in best_extracted_snippet:
        correct_matches[index] = best_extracted_snippet  # Store index and corrected text
        continue  # No mismatch

    # ❌ Otherwise, log mismatch and delete entry
    mismatched_indices.append(index)
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"❌ Mismatch in doc_id={doc_id}, entry {index} (Deleting entry)\n")
        log.write(f"   📌 Expected: {stored_snippet[:200]}...\n")
        log.write(f"   🔎 Best Extracted: {best_extracted_snippet[:200]}...\n")
        log.write(f"   🔢 Best Similarity: {best_similarity}%\n\n")

# Remove mismatched entries and reset index
df.drop(index=mismatched_indices, inplace=True)
df.reset_index(drop=True, inplace=True)  # ✅ Fix KeyError issue

# Update expected snippets with the best extracted version for correct matches
for index, best_snippet in correct_matches.items():
    if index < len(df):  # ✅ Ensure index exists
        try:
            old_references = json.loads(df.loc[index, "references"])
            for ref in old_references:
                ref["content"] = best_snippet  # ✅ Only update "content"
            df.loc[index, "references"] = json.dumps(old_references)
        except json.JSONDecodeError:
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"⚠️ WARNING: Failed to update references for entry {index} due to JSON error.\n")

# Save the cleaned dataset
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

# Final Summary
with open(LOG_FILE, "a", encoding="utf-8") as log:
    log.write("=" * 50 + "\n")
    log.write(f"✅ Verification completed: {len(df) + len(mismatched_indices)} entries checked.\n")
    log.write(f"⚠️ {len(mismatched_indices)} entries removed.\n")
    log.write(f"📂 Updated dataset saved to {OUTPUT_CSV}\n")

print(f"✅ Verification complete! Report saved to {LOG_FILE}")
print(f"⚠️ {len(mismatched_indices)} entries removed.")
print(f"📂 Updated dataset saved to {OUTPUT_CSV}")