import os.path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from constants import BASE_MODEL, PL_BERT_SINGLE_REPO_PATH
from util import extract_state, remove_state


def load_pl_bert_single():
    return AutoModelForSequenceClassification.from_pretrained(PL_BERT_SINGLE_REPO_PATH), AutoTokenizer.from_pretrained(
        PL_BERT_SINGLE_REPO_PATH)


if __name__ == "__main__":
    load_dotenv()
    tbert_single_path = os.path.expanduser(os.environ["TBERT_SINGLE_PATH"])

    # Create state dictionary for model.
    state_dict = torch.load(tbert_single_path, map_location=torch.device("cpu"))
    model_state_dict = extract_state(state_dict, "bert.")
    model_state_dict = remove_state(model_state_dict, "roberta.pooler")

    # Load model and inject state
    print("Loading model architecture with base weights...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
    print("Loading state dictionary into model")
    base_model.load_state_dict(model_state_dict, strict=True)
    print("Model was loaded successfully.")

    # Publish model
    print("Pushing models to hub.")
    base_model.push_to_hub(PL_BERT_SINGLE_REPO_PATH)
    tokenizer.push_to_hub(PL_BERT_SINGLE_REPO_PATH)
    print("Done.")
