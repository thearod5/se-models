import os.path

from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def publish_se_bert():
    se_bert_path = os.path.expanduser(os.environ["NL_BERT_PATH"])
    repo_path = "thearod5/nl-bert"

    model = AutoModelForSequenceClassification.from_pretrained(se_bert_path)
    tokenizer = AutoTokenizer.from_pretrained(se_bert_path)

    model.push_to_hub(repo_path)
    tokenizer.push_to_hub(repo_path)

    print("Done")


if __name__ == "__main__":
    load_dotenv()
    publish_se_bert()
