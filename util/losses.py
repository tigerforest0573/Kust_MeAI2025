# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from kornia.losses import BinaryFocalLossWithLogits

def build_loss_fn(config: dict) -> Tuple[nn.Module, nn.Module]:
    loss_name = config['name']
    if loss_name == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
        output_act = nn.Softmax(dim=-1)
    elif loss_name == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
        output_act = nn.Sigmoid()
    elif loss_name == "focal_loss":
        alpha = config.get('alpha', 0.25)
        gamma = config.get('gamma', 4.0)
        loss_fn = BinaryFocalLossWithLogits(
            alpha=alpha, 
            gamma=gamma, 
            reduction='mean'
        )
        output_act = nn.Sigmoid()
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")
    return loss_fn, output_act
