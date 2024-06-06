from collections import OrderedDict

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from transformers import AutoConfig

load_dotenv()

BASE_MODEL = "microsoft/codebert-base"


class TBertSiameseEncoder(SentenceTransformer):
    def __init__(self):
        config = AutoConfig.from_pretrained(BASE_MODEL)
        transformer = Transformer.models.Transformer(BASE_MODEL)
        pooling_layer = Pooling(word_embedding_dimension=config.hidden_size, pooling_mode_mean_tokens=True)
        modules = OrderedDict([
            ('encoder_model', transformer),
            ('pooling_model', pooling_layer)
        ])
        super().__init__(modules=modules)
