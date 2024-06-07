from typing import Dict

import torch
from torch import Tensor


def get_scores_from_logits(logits: Tensor):
    return torch.softmax(logits, 1).squeeze(dim=0)[1].item()


def extract_state(state_dict: Dict, prefix: str):
    if "." not in prefix:
        prefix += "."
    return {k.removeprefix(prefix): state_dict[k] for k in list(state_dict.keys()) if
            prefix in k}


def remove_state(state_dict: Dict, prefix: str):
    return {k: state_dict[k] for k in list(state_dict.keys()) if not k.startswith(prefix)}
