import os

import torch
from transformers import RobertaConfig

from pl_bert.models.pl_bert_siamese_cross_encoder import TBertSiameseCrossEncoder
from pl_bert.models.pl_bert_siamese_encoder import TBertSiameseEncoder
from util import extract_state


def load_tbert_siamese_encoder(state_path: str = None, state_prefix: str = "cbert."):
    if state_path is None:
        state_path = os.environ["STATE_PATH"]
    state_path = os.path.expanduser(state_path)
    model = TBertSiameseEncoder()

    # Load the custom weights
    state_dict = torch.load(state_path, map_location=torch.device('cpu'))
    embedding_model_keys = extract_state(state_dict, state_prefix)
    model.encoder_model.auto_model.load_state_dict(embedding_model_keys)

    return model


def load_tbert_siamese_cross_encoder(state_path: str = None, encoder_state_prefix: str = "cbert.", cls_state_prefix: str = "cls."):
    if state_path is None:
        state_path = os.environ["STATE_PATH"]
    state_path = os.path.expanduser(state_path)
    config = RobertaConfig()
    model = TBertSiameseCrossEncoder(config)

    # Load the custom weights
    state_dict = torch.load(state_path, map_location=torch.device('cpu'))
    encoder_state_dict = extract_state(state_dict, encoder_state_prefix)
    cls_state_dict = extract_state(state_dict, cls_state_prefix)
    model.classifier.load_state_dict(cls_state_dict)
    model.roberta.load_state_dict(encoder_state_dict)
    return model
