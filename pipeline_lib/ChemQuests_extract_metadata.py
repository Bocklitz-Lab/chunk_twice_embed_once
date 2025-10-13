# SPDX-License-Identifier: CC-BY-4.0

import json
import openai
import time
import os
import argparse

# Load OpenAI API Key securely
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set

# Paths to input/output JSON files
input_json_path = "./../hf/olmocr-out.jsonl"  # Original input file
modified_input_json_path = "./../hf/modified_olmocr-out.jsonl"  # New file with added doc_id
output_json_path = "./../hf/extracted_metadata.jsonl"  # Output file for extracted metadata
response_output_dir = "./../hf/openai_responses/"  # Directory to store raw OpenAI responses

# Ensure required directories exist
os.makedirs(response_output_dir, exist_ok=True)

# List of valid fields of study
VALID_FIELDS = {
    "Agriculture and Food Chemistry", "Analytical Chemistry", "Biological and Medicinal Chemistry",
    "Catalysis", "Chemical Education", "Chemical Engineering and Industrial Chemistry",
    "Earth, Space, and Environmental Chemistry", "Energy", "Inorganic Chemistry",
    "Materials Chemistry", "Materials Science", "Nanoscience", "Organic Chemistry",
    "Organometallic Chemistry", "Physical Chemistry", "Polymer Science",
    "Theoretical and Computational Chemistry"
}

# Function to call OpenAI API and extract metadata
def extract_metadata(text, doc_id, retries=3, backoff_factor=2):
    """ Calls OpenAI API to extract metadata with retry & backoff on errors. """

    print(f"\n[DEBUG] Extracting metadata for doc_id: {doc_id}")
    print(f"[DEBUG] Input text (truncated): {text[:200]}...")  # Show first 200 chars

    prompt = f"""
    Extract the following details from the given research paper:
    - Title
    - Abstract
    - Authors
    - Publication date
    - Fields of study (must match the provided valid categories)

    **Valid Fields of Study:**
    {", ".join(VALID_FIELDS)}

    Research Paper Content:
    {text[:5000]}  # Limit input to 5000 characters for efficiency

    Provide the response in JSON format with keys: title, abstract, authors, publication_date, fields_of_study.
    """

    for attempt in range(retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are an AI that extracts metadata from research papers."},
                          {"role": "user", "content": prompt}],
                temperature=0.2
            )

            response_content = response.choices[0].message.content

            print(f"[DEBUG] OpenAI Response (raw): {response_content[:200]}...")  # Truncate for readability

            # Save raw OpenAI response
            response_file = os.path.join(response_output_dir, f"{doc_id}.json")
            with open(response_file, "w", encoding="utf-8") as f:
                json.dump(response_content, f, indent=4)

            # Clean response (remove triple backticks if present)
            cleaned_response = response_content.strip().strip("```json").strip("```")

            try:
                extracted_data = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse cleaned OpenAI response for doc_id {doc_id}: {e}")
                return None

            # Ensure extracted fields of study are valid
            extracted_data["fields_of_study"] = [
                field for field in extracted_data.get("fields_of_study", []) if field in VALID_FIELDS
            ]

            return extracted_data  # Return extracted metadata if successful

        except openai.OpenAIError as e:
            print(f"[ERROR] API error on attempt {attempt + 1} for doc_id {doc_id}: {e}")
            time.sleep(backoff_factor ** attempt)  # Exponential backoff
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse OpenAI response for doc_id {doc_id}: {e}")
            return None  # Return None if JSON parsing fails

    return None  # Return None if all attempts fail

# Function to process JSONL file with offset support
def process_jsonl(offset):
    """ Processes JSONL file from the given offset and saves results incrementally. """
    processed_count = 0

    with open(input_json_path, "r", encoding="utf-8") as f_in, \
         open(modified_input_json_path, "w", encoding="utf-8") as f_mod, \
         open(output_json_path, "a", encoding="utf-8") as f_out:

        for i, line in enumerate(f_in):
            if i < offset:  # Skip lines until the offset is reached
                continue

            line = line.strip()
            if not line:  # Skip empty lines
                print(f"[WARNING] Skipping empty line {i}")
                continue  

            try:
                record = json.loads(line)  # Parse JSON
                text = record.get("text", "")

                # Assign a new doc_id based on line number
                doc_id = i  
                record["doc_id"] = doc_id  # Add doc_id to the original record

                # Write the modified input entry back to a new file
                f_mod.write(json.dumps(record) + "\n")

                print(f"\n[DEBUG] Processing doc_id: {doc_id} (Entry {i})")
                print(f"[DEBUG] Text length: {len(text)} characters")

                # Check if text is empty
                if not text.strip():
                    print(f"[WARNING] Skipping doc_id {doc_id}: Empty text field")
                    continue

                # Extract metadata using OpenAI API
                extracted_metadata = extract_metadata(text, doc_id)

                if extracted_metadata:
                    extracted_metadata["doc_id"] = doc_id  # Keep track of document ID
                    
                    # Write extracted metadata immediately to avoid data loss
                    f_out.write(json.dumps(extracted_metadata) + "\n")
                    f_out.flush()  # Ensure data is written to file

                processed_count += 1
                time.sleep(1)  # Avoid hitting rate limits

            except json.JSONDecodeError as e:
                print(f"[ERROR] Skipping invalid JSON line {i}: {e}")

    print(f"\n[INFO] Processing complete. {processed_count} new entries processed.")

# Add command-line argument support for offset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSONL dataset with OpenAI and extract metadata.")
    parser.add_argument("--offset", type=int, default=0, help="Start processing from this entry (default: 0)")
    args = parser.parse_args()

    process_jsonl(args.offset)