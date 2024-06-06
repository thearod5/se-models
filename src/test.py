import os.path

import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

load_dotenv()

if __name__ == "__main__":
    model_path = os.path.expanduser(os.environ["MODEL_PATH"])
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    embedding_model_keys = {k.removeprefix("cbert."): state_dict[k] for k in list(state_dict.keys()) if "cbert." in k}
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    model.load_state_dict(embedding_model_keys)
    print("Done")
