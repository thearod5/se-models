import torch
from torch import nn
from transformers import RobertaForSequenceClassification


class AvgPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class RelationClassifyHeader(nn.Module):
    """
    H2:
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.code_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)

        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, 2)

    def forward(self, code_hidden, text_hidden):
        pool_code_hidden = self.code_pooler(code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class TBertSiameseCrossEncoder(RobertaForSequenceClassification):
    def __init__(self, config):
        config.num_labels = 1
        super().__init__(config)
        self.classifier = RelationClassifyHeader(config)
