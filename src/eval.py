import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

from nl_bert.nl_bert_factory import load_nl_bert
from pl_bert.pl_bert_factory import load_tbert_siamese_cross_encoder, load_tbert_siamese_encoder
from util import get_scores_from_logits


def eval_siamese_encoder(model, texts):
    embeddings = model.encode(texts, convert_to_tensor=False)
    parent_embedding = embeddings[0:1]
    children_embeddings = embeddings[1:]

    # Compute cosine similarity
    sim_matrix = cosine_similarity(parent_embedding, children_embeddings)
    return sim_matrix


def eval_siamese_cross_encoder(model, texts):
    parent_text, child1_text, child2_text = texts
    score1 = model.predict(parent_text, child1_text)["scores"].item()
    score2 = model.predict(parent_text, child2_text)["scores"].item()
    scores = [round(s, 3) for s in [score1, score2]]
    sim_matrix = np.stack(scores)
    return sim_matrix


def eval_seq_classification(model, texts):
    parent_text, child1_text, child2_text = texts
    tokenizer = AutoTokenizer.from_pretrained("thearod5/nl-bert")
    payload1 = f"{parent_text}<sep>{child1_text}"
    payload2 = f"{parent_text}<sep>{child2_text}"

    input1 = tokenizer(payload1, padding=True, truncation=True, return_tensors="pt")
    input2 = tokenizer(payload2, padding=True, truncation=True, return_tensors="pt")

    output1 = model(**input1)
    scores1 = get_scores_from_logits(output1.logits)

    output2 = model(**input2)
    scores2 = get_scores_from_logits(output2.logits)

    return np.stack([scores1, scores2])


def eval(model: SentenceTransformer, evaluator):
    # Test
    parent_text = "Display Artifacts"
    child1_text = "A table view should be provided to display all project artifacts."
    child2_text = "The system should be able to."

    texts = [
        parent_text,
        child1_text,
        child2_text
    ]

    model.eval()
    sim_matrix = evaluator(model, texts)
    return sim_matrix


def run_evaluation():
    # Load model and tokenizer
    results = {}
    for model_name, (model_loader, model_evaluator) in model_registrar.items():
        model = model_loader()
        similarity_matrix = eval(model, model_evaluator)
        results[model_name] = similarity_matrix

    for model_name, sim_matrix in results.items():
        print("Model:", model_name)
        print("Similarity Matrix:", sim_matrix)


if __name__ == "__main__":
    model_registrar = {
        "nl-bert": (load_nl_bert, eval_seq_classification),
        "encoder": (load_tbert_siamese_encoder, eval_siamese_encoder),
        "cross_encoder": (load_tbert_siamese_cross_encoder, eval_siamese_cross_encoder),
    }
    run_evaluation()
