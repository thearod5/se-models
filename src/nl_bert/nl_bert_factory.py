from transformers import AutoModelForSequenceClassification, AutoTokenizer

from constants import NL_BERT_REPO_PATH


def load_nl_bert():
    return AutoModelForSequenceClassification.from_pretrained(NL_BERT_REPO_PATH), AutoTokenizer.from_pretrained(NL_BERT_REPO_PATH)
