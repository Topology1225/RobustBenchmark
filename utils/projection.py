import torch


class Linf_clamper(object):
    def __init__(self, epsilon, lower, upper, device):
        self.epsilon = epsilon
        self.lower   = lower.to(device)
        self.upper   = upper.to(device)

    def __call__(self, x):
        clamped_x = torch.clamp(
            input=x,
            min=self.lower,
            max=self.upper   
        )
        return clamped_x
    
    def __del__(self):
        del self.lower
        del self.upper


def get_projection(norm:str, *args, **kwargs):
    if norm=="Linf":
        return Linf_clamper(
            *args, **kwargs
        )

    elif norm=="L2":
        raise NotImplementedError
    else:
        raise ValueError(f"{norm} is not supported.")
        