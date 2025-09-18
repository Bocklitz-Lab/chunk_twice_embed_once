import mteb
import os
from tqdm import tqdm
import wandb
import json
import time
from transformers import AutoModel, AutoTokenizer  # ✅ Load models using Hugging Face
import os
from sentence_transformers import SentenceTransformer, models
from mteb.evaluation.evaluators import RetrievalEvaluator  # ✅ Correct import
import sys
from mteb.tasks.Retrieval.eng.FSUChemRxivQest import FSUChemRxivQest

# Ensure the correct MTEB path is added
sys.path.insert(0, os.path.abspath("./mteb"))

import mteb
print("Using local MTEB version from:", mteb.__file__)

# from huggingface_hub import login
# login()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# def is_run_available(model_name, model_revision):
#     api = wandb.Api()
#     runs = api.runs('Chembedding - Benchmarking2')
#     for run in runs:
#         if run.name == model_name and run.config['revision'] == model_revision and run.state == "finished":
#             return True
#     return False

def is_run_available(model_name, model_revision):
    api = wandb.Api()
    runs = api.runs('Chembedding - Benchmarking2')

    print(f"🔍 Checking previous runs for {model_name} - {model_revision}")

    for run in runs:
        print(f"   ➡ Found run: {run.name}, Revision: {run.config.get('revision', 'N/A')}, State: {run.state}")

        if run.name == model_name and run.config.get('revision') == model_revision and run.state == "finished":
            print(f"✅ Skipping {model_name} - {model_revision} (Already finished)")
            return True  # Run already exists

    print(f"🚀 Running {model_name} - {model_revision} (Not found in W&B)")
    return False  # No completed run found



def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def json_parser(data):
    task_name = data["task_name"]
    output = {}
    if task_name.endswith("PC"):
        output["PairClassification (Max F1)"] = data["scores"]["test"][0]["main_score"]
    elif task_name.endswith("Classification"):
        output["Classification (Accuracy)"] = data["scores"]["test"][0]["main_score"]
    elif "BitextMining" in task_name or task_name.endswith("BM"):
        output["Bitext Mining (F1)"] = data["scores"]["test"][0]["main_score"]
    elif task_name.endswith("Retrieval"):
        output["Retrieval (NDCG@10)"] = data["scores"]["test"][0]["main_score"]
    elif task_name.endswith("Clustering"):
        output["Clustering (V Measure)"] = data["scores"]["test"][0]["main_score"]
    return output

if __name__ == "__main__":
    now = time.time()

    models = {
                "sentence-transformers/bert-base-nli-mean-tokens": "no_revision_available",
                "allenai/scibert_scivocab_uncased": "no_revision_available",
                "nomic-ai/nomic-bert-2048": "40b98394640e630d5276807046089b233113aa87",
                "intfloat/e5-small": "e272f3049e853b47cb5ca3952268c6662abda68f",
                "intfloat/e5-base": "b533fe4636f4a2507c08ddab40644d20b0006d6a",
                "intfloat/e5-large": "4dc6d853a804b9c8886ede6dda8a073b7dc08a81",
                "intfloat/e5-small-v2": "dca8b1a9dae0d4575df2bf423a5edb485a431236",
                "intfloat/e5-base-v2": "1c644c92ad3ba1efdad3f1451a637716616a20e8",
                "intfloat/e5-large-v2": "b322e09026e4ea05f42beadf4d661fb4e101d311",
                "intfloat/multilingual-e5-small": "fd1525a9fd15316a2d503bf26ab031a61d056e98",
                "intfloat/multilingual-e5-base": "d13f1b27baf31030b7fd040960d60d909913633f",
                "intfloat/multilingual-e5-large": "ab10c1a7f42e74530fe7ae5be82e6d4f11a719eb",
                "nomic-ai/nomic-embed-text-v1": "0759316f275aa0cb93a5b830973843ca66babcf5",
                "nomic-ai/nomic-embed-text-v1.5": "b0753ae76394dd36bcfb912a46018088bca48be0",
                "recobo/chemical-bert-uncased": "no_revision_available",
                "BAAI/bge-m3": "no_revision_available",
                "BAAI/bge-small-en": "no_revision_available",
                "BAAI/bge-base-en": "no_revision_available",
                "BAAI/bge-large-en": "no_revision_available",
                "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
                "BAAI/bge-large-en-v1.5": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
                "sentence-transformers/all-mpnet-base-v2": "no_revision_available",
                "sentence-transformers/multi-qa-mpnet-base-dot-v1": "no_revision_available",
                "sentence-transformers/all-MiniLM-L12-v2": "no_revision_available",
                "sentence-transformers/all-MiniLM-L6-v2": "no_revision_available",
                "m3rg-iitd/matscibert": "no_revision_available",
                "allenai/specter": "no_revision_available",
                "facebook/contriever": "no_revision_available",
                "hkunlp/instructor-xl": "no_revision_available",
                "facebook/contriever-msmarco": "no_revision_available",
                "sentence-transformers/gtr-t5-base": "no_revision_available",
                "sentence-transformers/gtr-t5-large": "no_revision_available",
                "sentence-transformers/gtr-t5-xl": "no_revision_available",
                "jinaai/jina-embeddings-v2-small-en": "no_revision_available",
                "jinaai/jina-embeddings-v2-base-en": "no_revision_available",
                "jinaai/jina-embeddings-v2-large-en": "no_revision_available",
                "microsoft/mpnet-base": "no_revision_available",
                "microsoft/MiniLM-L12-H384-uncased": "no_revision_available",
                "sentence-transformers/all-MiniLM-L6-v2": "no_revision_available",
                "sentence-transformers/all-MiniLM-L12-v2": "no_revision_available",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "79f2382ceacceacdf38563d7c5d16b9ff8d725d6",
                "nomic-ai/nomic-embed-text-v2-moe": "no_revision_available",
                "BAAI/bge-multilingual": "no_revision_available",
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract": "no_revision_available",
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "no_revision_available",
                "recobo/chemical-bert-uncased": "no_revision_available",
                "m3rg-iitd/matscibert": "no_revision_available",
                "DeepChem/ChemBERTa-77M-MTR": "no_revision_available",
                "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract": "no_revision_available",
                # "describeai/gemini": "no_revision_available",
            }

    all_tasks = [
        # Retrieval 
        "ChemNQRetrieval",
        "ChemHotpotQARetrieval",
        # "CoconutRetrieval",
        "FSUChemRxivQest"
    ]

    # Get built-in tasks (expects task names)
    built_in_tasks = mteb.get_tasks(tasks=all_tasks)

    # Initialize custom task separately
    custom_tasks = [FSUChemRxivQest()]

    # Combine both
    # Get built-in tasks (returns a tuple)
    built_in_tasks = mteb.get_tasks(tasks=all_tasks)

    # Convert tuple to list and add custom task
    tasks = list(built_in_tasks) + [FSUChemRxivQest()]






    for model_full_name, model_rev in tqdm(models.items()):
        model_name = model_full_name.split("/")[-1]

        # if is_run_available(model_name, model_rev):
        #     print(f"Skipping {model_name} - {model_rev}")
        #     continue

        try:
            wandb.init(project='Chembedding - Benchmarking2', name=model_name, config={"revision": model_rev})

            # ✅ Authenticate with Hugging Face for private models
            tokenizer = AutoTokenizer.from_pretrained(model_full_name, token=True, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_full_name, token=True, trust_remote_code=True)

            
            # ✅ Use correct get_model() syntax
            wrapped_model = mteb.get_model(model_full_name)

            # Run evaluation
            evaluation = mteb.MTEB(tasks=tasks)
            evaluation.run(wrapped_model, output_folder="chem_results", overwrite_results=False)


        except Exception as e:
            print(f"Error Evaluating Model {model_name}: {e}")
            wandb.finish()
            continue

        # ✅ Logging results to wandb
        for task_name in tqdm(all_tasks):
            json_path = os.path.join("chem_results", model_full_name.replace("/", "__"), model_rev, f"{task_name}.json")
            if not os.path.exists(json_path):
                print(f"Skipping missing results for {task_name} - {model_name}")
                continue

            data = read_json(json_path)
            output = json_parser(data)
            wandb.log(output)

            for metric, score in output.items():
                table = wandb.Table(data=[[metric, score]], columns=["Metric", "Score"])
                bar_plot = wandb.plot.bar(table, "Metric", "Score", title=f"{task_name} Performance")
                wandb.log({f"{task_name}_bar_plot": bar_plot})

        wandb.finish()

    # ✅ Print elapsed time in HH:MM:SS format
    elapsed = time.time() - now
    print(f"Elapsed time: {int(elapsed // 3600)} hours, {int((elapsed % 3600) // 60)} minutes, {int(elapsed % 60)} seconds")