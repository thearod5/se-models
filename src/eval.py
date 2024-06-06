from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from factory import load_tbert_cross_encoder, load_tbert_encoder


def eval(model: SentenceTransformer):
    # Test
    parent_text = "Display Artifacts"
    child1_text = "A table view should be provided to display all project artifacts."
    child2_text = "The system should be able to generate documentation for a set of artifacts."
    texts = [
        parent_text,
        child1_text,
        child2_text
    ]

    if hasattr(model, "encoder"):
        embeddings = model.encode(texts, convert_to_tensor=False)
    else:
        embeddings = model.predict([parent_text, child1_text], [parent_text, child2_text])

    parent_embedding = embeddings[0:1]
    children_embeddings = embeddings[1:]

    # Compute cosine similarity
    sim_matrix = cosine_similarity(parent_embedding, children_embeddings)
    return sim_matrix


if __name__ == "__main__":
    # Load model and tokenizer
    model_name = "cross_encoder"
    model_names = {
        "encoder": load_tbert_encoder,
        "cross_encoder": load_tbert_cross_encoder
    }
    model = model_names[model_name]()

    # Evaluate model
    similarity_matrix = eval(model)

    # Log job finished.
    print("Similarity Matrix:", similarity_matrix)
