import os

from dotenv import load_dotenv

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
    repo_name = "thearod5/tbert-encoder"  # Change this to your Hugging Face model repo name

    # Load model and tokenizer
    model, tokenizer = load_embedding_model(model_path)

    # Save model and tokenizer
    model.push_to_hub("tbert-encoder")

    # Log job finished.
    print("Done.")
