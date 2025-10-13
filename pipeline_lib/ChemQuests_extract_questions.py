# SPDX-License-Identifier: CC-BY-4.0

import json
import openai
import time
import os
import argparse

# Load OpenAI API Key securely
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set

# Paths to input/output JSON files
input_json_path = "./../hf/olmocr-out.jsonl"  # Input file with research paper texts
output_json_path = "./../hf/generated_questions.jsonl"  # Output file for questions
response_output_dir = "./../hf/openai_question_responses/"  # Directory to store raw OpenAI responses

# Ensure response output directory exists
os.makedirs(response_output_dir, exist_ok=True)

# Function to call OpenAI API and generate chemistry questions
def generate_questions(text, doc_id, retries=3, backoff_factor=2):
    """ Calls OpenAI API to generate chemistry-related questions from research papers. """

    print(f"\n[DEBUG] Generating questions for doc_id: {doc_id}")
    print(f"[DEBUG] Input text (truncated): {text[:200]}...")  # Show first 200 chars

    prompt = f"""
    Based on the following research paper, generate 10 **chemistry-related** questions. 
    The questions should be specific to the **scientific content** (e.g., chemical properties, reactions, mechanisms), 
    NOT about the general purpose or methods of the paper.
    
    For each question, provide:
    - The **question**.
    - The **correct answer**.
    - A **relevant part of the text** where the answer is found.

    **Research Paper Content (truncated for processing efficiency):**
    {text[:20000]}  # Limit input to 5000 characters

    Return the response in **JSON format** as a list of dictionaries:
    ```json
    [
        {{"question": "...", "answer": "...", "text_snippet": "..."}},
        {{"question": "...", "answer": "...", "text_snippet": "..."}}
    ]
    ```
    """

    for attempt in range(retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are an AI assistant generating chemistry-related questions from research papers."},
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
                questions_data = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse cleaned OpenAI response for doc_id {doc_id}: {e}")
                return None

            return questions_data  # Return extracted questions if successful

        except openai.OpenAIError as e:
            print(f"[ERROR] API error on attempt {attempt + 1} for doc_id {doc_id}: {e}")
            time.sleep(backoff_factor ** attempt)  # Exponential backoff
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse OpenAI response for doc_id {doc_id}: {e}")
            return None  # Return None if JSON parsing fails

    return None  # Return None if all attempts fail

# Function to process JSONL file and generate questions
def process_jsonl(offset):
    """ Processes JSONL file from the given offset and saves results incrementally. """
    processed_count = 0

    with open(input_json_path, "r", encoding="utf-8") as f_in, open(output_json_path, "a", encoding="utf-8") as f_out:
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

                print(f"\n[DEBUG] Processing doc_id: {doc_id} (Entry {i})")
                print(f"[DEBUG] Text length: {len(text)} characters")

                # Check if text is empty
                if not text.strip():
                    print(f"[WARNING] Skipping doc_id {doc_id}: Empty text field")
                    continue

                # Generate questions using OpenAI API
                questions_data = generate_questions(text, doc_id)

                if questions_data:
                    output_entry = {
                        "doc_id": doc_id,
                        "questions": questions_data
                    }

                    # Write extracted metadata immediately to avoid data loss
                    f_out.write(json.dumps(output_entry) + "\n")
                    f_out.flush()  # Ensure data is written to file

                processed_count += 1
                time.sleep(1)  # Avoid hitting rate limits

            except json.JSONDecodeError as e:
                print(f"[ERROR] Skipping invalid JSON line {i}: {e}")

    print(f"\n[INFO] Processing complete. {processed_count} new entries processed.")

# Add command-line argument support for offset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSONL dataset with OpenAI and generate chemistry-related questions.")
    parser.add_argument("--offset", type=int, default=0, help="Start processing from this entry (default: 0)")
    args = parser.parse_args()

    process_jsonl(args.offset)