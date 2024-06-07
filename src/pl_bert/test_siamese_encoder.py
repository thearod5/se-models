from sentence_transformers import SentenceTransformer

from eval import eval

if __name__ == "__main__":
    repo_name = "thearod5/tbert-siamese-encoder"
    model = SentenceTransformer(repo_name)

    similarity_matrix = eval(model)
    print("Similarity Matrix:", similarity_matrix)
