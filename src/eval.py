import torch
from sklearn.metrics.pairwise import cosine_similarity


def eval(model, tokenizer):
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


if __name__ == "__main__":
    pass
