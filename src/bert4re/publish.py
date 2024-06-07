import os

from dotenv import load_dotenv
from transformers import AutoModelForMaskedLM


def bert4re():
    bert4re_path = os.path.expanduser(os.environ["BERT4RE_PATH"])
    model_path = os.path.expanduser(bert4re_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model.push_to_hub("thearod5/bert4re")
    print("Done")


if __name__ == '__main__':
    load_dotenv()
    bert4re()
