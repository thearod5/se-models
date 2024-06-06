import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models import load_embedding_model


def eval(model: SentenceTransformer, tokenizer):
    # Test
    texts = [
        "Display Artifacts",
        "A table view should be provided to display all project artifacts.",
        "The system should be able to generate documentation for a set of artifacts."
    ]
    embeddings = model.encode(texts, convert_to_tensor=False)

    parent_embedding = embeddings[0:1]
    children_embeddings = embeddings[1:]

    # Compute cosine similarity
    sim_matrix = cosine_similarity(parent_embedding, children_embeddings)
    return sim_matrix


if __name__ == "__main__":
    model_path = os.path.expanduser(os.environ["MODEL_PATH"])

    # Load model and tokenizer
    model, tokenizer = load_embedding_model(model_path)

    # Evaluate model
    similarity_matrix = eval(model, tokenizer)

    # Log job finished.
    print("Similarity Matrix:", similarity_matrix)
