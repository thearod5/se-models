import torch
from torch import Tensor


def get_scores_from_logits(logits: Tensor):
    return torch.softmax(logits, 1).squeeze(dim=0)[1].item()
