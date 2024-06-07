from transformers import AutoModelForSequenceClassification


def load_nl_bert():
    return AutoModelForSequenceClassification.from_pretrained("thearod5/nl-bert")
