import os

from dotenv import load_dotenv
from transformers import AutoModelForMaskedLM, AutoTokenizer


def bert4re():
    bert4re_path = os.path.expanduser(os.environ["BERT4RE_PATH"])
    repo_path = "thearod5/bert4re"

    model = AutoModelForMaskedLM.from_pretrained(bert4re_path)
    tokenizer = AutoTokenizer.from_pretrained(bert4re_path)

    model.push_to_hub(repo_path)
    tokenizer.push_to_hub(repo_path)
    print("Done")


if __name__ == '__main__':
    load_dotenv()
    bert4re()
