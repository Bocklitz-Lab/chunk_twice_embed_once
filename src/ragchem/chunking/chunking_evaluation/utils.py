from enum import Enum
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
from chromadb.utils import embedding_functions
import tiktoken

def find_query_despite_whitespace(document, query):

    # Normalize spaces and newlines in the query
    normalized_query = re.sub(r'\s+', ' ', query).strip()
    
    # Create a regex pattern from the normalized query to match any whitespace characters between words
    pattern = r'\s*'.join(re.escape(word) for word in normalized_query.split())
    
    # Compile the regex to ignore case and search for it in the document
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)
    
    if match:
        return document[match.start(): match.end()], match.start(), match.end()
    else:
        return None
    

import unicodedata
import re
from fuzzywuzzy import process, fuzz

def normalize_text(text):
    """Normalize Unicode text to NFKC form for consistent comparison."""
    return unicodedata.normalize("NFKC", text)

def rigorous_document_search(document: str, target: str):
    """
    Searches for a target string within a document, handling Unicode normalization, whitespace variations, 
    and fuzzy matching for approximate matches.
    
    Args:
        document (str): The document to search within.
        target (str): The text string to find.

    Returns:
        tuple: (best_match, start_index, end_index) if found, otherwise None.
    """
    if not document or not target:
        return None  # Ensure inputs are valid
    
    # Normalize both document and target
    document = normalize_text(document)
    target = normalize_text(target)
# In your evaluation or data prep code:
    document = document.replace("�", "")
    target = target.replace("�", "")

    # Remove trailing period from target (common in chunk searches)
    target = target.rstrip('.')

    # 1️⃣ Exact Match Search
    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index

    # 2️⃣ Whitespace-Insensitive Search
    raw_search = find_query_despite_whitespace(document, target)
    if raw_search is not None:
        return raw_search

    # 3️⃣ Fuzzy Matching for Approximate Searches
    sentences = re.split(r'[.!?]\s*|\n', document)  # Split into sentences
    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)

    if best_match and best_match[1] >= 95:  # Adjusted threshold for flexibility
        reference = best_match[0]
        start_index = document.find(reference)
        end_index = start_index + len(reference)
        return reference, start_index, end_index

    # 4️⃣ No match found
    return None


def get_openai_embedding_function():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key is None:
        raise ValueError("You need to set an embedding function or set an OPENAI_API_KEY environment variable.")
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name="text-embedding-3-large"
    )
    return embedding_function

# Count the number of tokens in each page_content
def openai_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

class Language(str, Enum):
    """Enum of the programming languages."""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"