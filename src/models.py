import os

import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig, RobertaConfig

load_dotenv()


class TBertEncoderConfig(PretrainedConfig):
    model_type = "tbert_encoder"

    def __init__(self, base_model_config=RobertaConfig(), base_model_path: str = "microsoft/codebert-base", **kwargs):
        super().__init__(**kwargs)
        self.base_model_config = base_model_config
        self.base_model_path = base_model_path


class TBertEncoder(PreTrainedModel):
    config_class = TBertEncoderConfig

    def __init__(self, config: TBertEncoderConfig):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config.base_model_path)
        self.pooler_hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        embeddings = self.pooler(outputs).view(-1, self.hidden_size)
        return embeddings


def load_embedding_model(model_path: str, base_model_path: str):
    model_path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = TBertEncoder(TBertEncoderConfig())

    # Load the custom weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    embedding_model_keys = {k.removeprefix("cbert."): state_dict[k] for k in list(state_dict.keys()) if "cbert." in k}
    model.model.load_state_dict(embedding_model_keys)

    return model, tokenizer
