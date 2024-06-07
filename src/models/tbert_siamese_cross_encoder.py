import torch
from torch import nn
from torch.nn import MSELoss
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, RobertaConfig

from constants import BASE_MODEL


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

    def forward(self, a_hidden, b_hidden):
        pool_code_hidden = self.code_pooler(a_hidden[0])
        pool_text_hidden = self.text_pooler(b_hidden[0])
        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class TBertSiameseCrossEncoder(PreTrainedModel):
    config_class = RobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.classifier = RelationClassifyHeader(config)
        self.roberta = AutoModel.from_pretrained(BASE_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def forward(self, a_input, b_input, labels=None):
        a_hidden_state = self.roberta(**a_input)
        b_hidden_state = self.roberta(**b_input)
        output = self.classifier(a_hidden_state, b_hidden_state)
        sim_scores = torch.softmax(output, 1).squeeze(dim=0)[1]
        output_dict = {"scores": sim_scores}
        if labels is not None:
            loss_fct = MSELoss()
            rel_loss = loss_fct(sim_scores, labels)
            output_dict['loss'] = rel_loss
        return output_dict

    def predict(self, text_a: str, text_b: str):
        a_input = self.tokenizer(text_a, return_tensors="pt", padding=True, truncation=True)
        b_input = self.tokenizer(text_b, return_tensors="pt", padding=True, truncation=True)
        output = self.forward(a_input, b_input)
        return output
