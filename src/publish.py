import os

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from sklearn.metrics.pairwise import cosine_similarity

from models import load_embedding_model

load_dotenv()


def create_model_card():
    card = """
    # Model Card for tbert-embedding
    
    ## Model Description
    This repository contains the embedding model used in the siamese architecture described in the paper: "Traceability Transformed: Generating More Accurate Links with Pre-Trained BERT Models". The model was a top performer in generating traceability links between software artifacts. The original architecture utilized a relational classifier to create similarity scores between text pairs, resembling a cross-encoder. This model extracts the embedding component to enable the use of cosine similarity between embeddings, thus speeding up the process.
    
    **Note:** Evaluation is pending review.
    
    ## Intended Uses & Limitations
    This model is intended for producing embeddings of software artifacts for purposes such as clustering, Retrieval-Augmented Generation (RAG), or creating traceability links via similarity measures like cosine similarity.
    
    ## Training, Evaluation, and Results
    For detailed information on training, evaluation, and results, refer to the original [paper](https://arxiv.org/pdf/2102.04411).
    
    ## Citation
    ```
    @misc{lin2021traceability,
      title={Traceability Transformed: Generating More Accurate Links with Pre-Trained BERT Models}, 
      author={Jinfeng Lin and Yalin Liu and Qingkai Zeng and Meng Jiang and Jane Cleland-Huang},
      year={2021},
      eprint={2102.04411},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
    }
    ```
    """
    return card


if __name__ == "__main__":
    model_path = os.path.expanduser(os.environ["MODEL_PATH"])
    base_model_path = "microsoft/codebert-base"
    repo_name = "thearod5/tbert-encoder"  # Change this to your Hugging Face model repo name

    # Load model and tokenizer
    model, tokenizer = load_embedding_model(model_path, base_model_path)

    # Save model and tokenizer
    model.save_pretrained(repo_name)
    tokenizer.save_pretrained(repo_name)

    # Create a model card
    model_card = create_model_card()
    with open(os.path.join(repo_name, "README.md"), "w") as f:
        f.write(model_card)

    # Publish to Hugging Face
    api = HfApi()
    username = HfFolder.get_token().split('-')[0]  # Assumes token is in the format 'username-xxxxxx'

    # Create repo if it does not exist
    create_repo(repo_id=repo_name, repo_type="model", token=HfFolder.get_token())

    # Push to Hugging Face
    repo = Repository(local_dir=repo_name, clone_from=repo_name, use_auth_token=True)
    repo.push_to_hub(commit_message="Initial commit of the model")

    print("Model published to Hugging Face.")

    # Test
    texts = [
        "Display Artifacts",
        "A table view should be provided to display all project artifacts.",
        "The system should be able to generate documentation for a set of artifacts."
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    # Embeddings
    with torch.no_grad():
        embeddings = model(**inputs)

    parent_embedding = embeddings[0:1]
    children_embeddings = embeddings[1:]

    # Compute cosine similarity
    score = cosine_similarity(parent_embedding, children_embeddings)
    print("Done", score)
