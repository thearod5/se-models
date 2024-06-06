import os
from collections import OrderedDict

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from transformers import AutoConfig, AutoTokenizer

load_dotenv()

BASE_MODEL = "microsoft/codebert-base"


class TBertEncoder(SentenceTransformer):
    def __init__(self):
        config = AutoConfig.from_pretrained(BASE_MODEL)
        transformer = Transformer(BASE_MODEL)
        pooling_layer = Pooling(word_embedding_dimension=config.hidden_size, pooling_mode_mean_tokens=True)
        modules = OrderedDict([
            ('encoder_model', transformer),
            ('pooling_model', pooling_layer)
        ])
        super().__init__(modules=modules)


def load_tbert_encoder(state_path: str = None, base_model_path: str = BASE_MODEL, state_prefix: str = "cbert."):
    if state_path is None:
        state_path = os.environ["STATE_PATH"]
    state_path = os.path.expanduser(state_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = TBertEncoder()

    # Load the custom weights
    state_dict = torch.load(state_path, map_location=torch.device('cpu'))
    embedding_model_keys = {k.removeprefix(state_prefix): state_dict[k] for k in list(state_dict.keys()) if state_prefix in k}
    model.encoder_model.auto_model.load_state_dict(embedding_model_keys)

    return model, tokenizer
