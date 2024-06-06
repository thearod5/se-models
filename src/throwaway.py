from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.push_to_hub("thearod5/replica")
