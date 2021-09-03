import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseCriterion(nn.Module):
    def __init__(self, targeted:bool=False):
        super().__init__()
        self.targeted = targeted
    
    def forward(self, x, y):
        raise NotImplementedError


class CrossEnropy(BaseCriterion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, y_pred, y_true):
        loss = F.cross_entropy(
            input=y_pred,
            target=y_true
        )
        return loss

        


def get_criterion(criterion_name:str):
    if criterion_name == "CrossEnropy":
        return CrossEnropy()