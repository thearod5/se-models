import os
from collections import OrderedDict

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, RobertaConfig

load_dotenv()


class TBertEncoderConfig(RobertaConfig):
    model_type = "tbert_encoder"

    def __init__(self, base_model_path: str = "microsoft/codebert-base", **kwargs):
        super().__init__(**kwargs)
        self.base_model_path = base_model_path


import torch
from torch import nn
from transformers import RobertaModel, RobertaTokenizer


class CustomRobertaModel(nn.Module):
    def __init__(self, model_name: str):
        super(CustomRobertaModel, self).__init__()
        self.transformer = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

    def forward(self, batch_encoding, **kwargs):
        input_ids = batch_encoding["input_ids"]
        attention_mask = batch_encoding["attention_mask"]
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def save(self, path):
        self.transformer.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path):
        model = cls.__new__(cls)
        model.transformer = RobertaModel.from_pretrained(path)
        model.tokenizer = RobertaTokenizer.from_pretrained(path)
        return model

    def tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").data


class AveragePooling(nn.Module):
    def __init__(self, hidden_size: int):
        super(AveragePooling, self).__init__()
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, hidden_size))
        self.hidden_size = hidden_size

    def forward(self, batch_tensor, **kwargs):
        embeddings = self.pooler(batch_tensor).view(-1, self.hidden_size)
        return {
            "sentence_embedding": embeddings
        }

    def save(self, path):
        # Save pooling configuration if needed
        pass

    @classmethod
    def load(cls, path):
        return cls()


class TBertEncoder(SentenceTransformer):
    def __init__(self):
        custom_roberta = CustomRobertaModel("microsoft/codebert-base")
        pooling_layer = AveragePooling(custom_roberta.transformer.config.hidden_size)
        modules = OrderedDict([
            ('encoder_model', custom_roberta),
            ('pooling_model', pooling_layer)
        ])
        super().__init__(modules=modules)


def load_embedding_model(model_path: str, base_model_path: str = "microsoft/codebert-base"):
    model_path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = TBertEncoder()

    # Load the custom weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    embedding_model_keys = {k.removeprefix("cbert."): state_dict[k] for k in list(state_dict.keys()) if "cbert." in k}
    model.encoder_model.transformer.load_state_dict(embedding_model_keys)

    return model, tokenizer
