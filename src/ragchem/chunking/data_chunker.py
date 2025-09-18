import os
import json
from chunking_evaluation.chunking import RecursiveTokenChunker
from chunking_evaluation.utils import openai_token_count

# Folder paths
input_folder = "./chunking_evaluation/evaluation_framework/general_evaluation_data/corpora"
output_folder = "./chunked_texts"
os.makedirs(output_folder, exist_ok=True)
overlap_size = 0
# Initialize RecursiveTokenChunker
chunker = RecursiveTokenChunker(chunk_size=100, chunk_overlap=overlap_size, length_function=openai_token_count)

# Process all text files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):  # Process only text files
        file_path = os.path.join(input_folder, filename)

        # Read the text file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Chunk the text
        chunks = chunker.split_text(text)

        # Store chunks with their start and end indices
        chunked_data = []
        start_index = 0  # Initialize start index

        for chunk in chunks:
            end_index = start_index + len(chunk)  # Calculate end index

            chunked_data.append({
                "document_name": filename,  # Store document name
                "chunk_text": chunk,
                "start_index": start_index,
                "end_index": end_index
            })

            start_index = end_index - overlap_size  # Adjust for overlap

        # Save to JSON
        output_path = os.path.join(output_folder, f"{filename}.json")
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(chunked_data, out_file, indent=4)

        print(f"Processed {filename}: {len(chunked_data)} chunks saved.")

print("Chunking complete! All files processed.")
