import os.path

from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from constants import NL_BERT_REPO_PATH


def publish_nl_bert():
    nl_bert_path = os.path.expanduser(os.environ["NL_BERT_PATH"])

    model = AutoModelForSequenceClassification.from_pretrained(nl_bert_path)
    tokenizer = AutoTokenizer.from_pretrained(nl_bert_path)

    model.push_to_hub(NL_BERT_REPO_PATH)
    tokenizer.push_to_hub(NL_BERT_REPO_PATH)

    print("Done")


if __name__ == "__main__":
    load_dotenv()
    publish_nl_bert()
