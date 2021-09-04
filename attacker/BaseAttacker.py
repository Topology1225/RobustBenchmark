import datetime

import torch
from torch.nn.parallel import DataParallel as DP

from utils.projection import get_projection
from utils.criterion  import get_criterion

class BaseAttacker(object):
    iteration=1
    def __init__(self, verbose:bool=True):
        super().__init__()
        self.verbose=verbose

        start_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S") 
        msg = f"Robust BenchMark Start at {start_time}"
        print(msg)
    
    def build(
        self, epsilon:float, model, criterion:str,
        norm:str, device_ids=[], *args, **kwargs):
        self.epsilon = epsilon
        self.set_device(device_ids)
        self.device_ids = device_ids
        self.norm = norm
        
        if len(device_ids) >= 2:
            self.model = DP(model, device_ids=device_ids)
        else:
            self.model = model.to(self.device)

        self.criterion = get_criterion(
            criterion_name=criterion
        )
        self._build(*args, **kwargs)

    
    def set_device(self, device_ids):
        if len(device_ids) == 0:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{min(device_ids)}")
        
        self.device = device

    def attack(self, x_true, y_true, *args, **kwargs):
        self.set_bound(x_true, self.norm)

        #Linf only
        self.projection = get_projection(
            norm=self.norm, epsilon=self.epsilon,
            lower=self.lower, upper=self.upper, device=self.device
            )

        print("Iteration   Objective   Best Objective. Success.")

        self._attack(x=x_true, y=y_true, *args, **kwargs)
    
    def set_bound(self, x, norm):
        if norm=="Linf":
            self.lower = torch.clamp(
                input=x - self.epsilon * torch.ones_like(x),
                min=0, max=1
            )
            self.upper = torch.clamp(
                input=x + self.epsilon * torch.ones_like(x),
                min=0, max=1
            )
            
        elif norm=="L2":
            pass

        else:
            raise ValueError(f"{norm} is not supported.")


    def do_verbose(self, iteration, loss, best_loss, success):
        if not self.verbose:
            return 0

        if not isinstance(loss, float):
            loss = loss.item()

        msg = (
            f"{iteration}".center(9, " "),
            f"  ",
            f"{loss:.3f}".center(9, " "),
            f"  ",
            f"{best_loss:.3f}".center(18, " "),
            f"{success}".center(7, " ")
        )
        print("".join(msg))

    def _attack(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_initialpoint(self, x):
        # random
        width = (self.upper - self.lower)
        perturb = torch.rand(self.upper.shape)
        x_init = width * perturb + self.lower
        
        return x_init

    @torch.enable_grad()
    def get_grad(self, xk, y):
        xk = xk.requires_grad_(True)
        logits = self.model(xk)
        loss_indiv   = self.criterion(
            logits, y
        )

        grad = torch.autograd.grad(
            loss_indiv, xk
        )[0]

        return grad, loss_indiv, logits

    def __del__(self, *args, **kwargs):
        end_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S") 
        msg = f"Robust BenchMark End at {end_time}"
        print(msg)


