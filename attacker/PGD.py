import torch

from .BaseAttacker import BaseAttacker


class PGDAttacker(BaseAttacker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # hardcode
    
    def _build(self, eta, iteration):
        self.eta       = eta
        self.iteration = 100

    @torch.no_grad()
    def _attack(self, x, y, *args, **kwargs):

        bs = x.shape[0]
        xk = self.get_initialpoint(x.clone())

        xk, y = xk.to(self.device), y.to(self.device)

        grad, loss_indiv, logits = self.get_grad(xk=xk, y=y)

        best_loss_indiv = loss_indiv.cpu().clone()
        best_logits     = logits.cpu().clone()

        for iteration in range(self.iteration):

            # 更新
            if self.norm == "Linf":
                xk += self.epsilon * torch.sign(grad)

            xk = self.projection(xk)

            grad, loss_indiv, logits = self.get_grad(xk=xk, y=y)

            best_loss_index = (loss_indiv.cpu() > best_loss_indiv)
            best_loss_indiv[best_loss_index] = loss_indiv[best_loss_index].cpu().clone()
            best_logits[best_loss_index]     = logits[best_loss_index].cpu().clone()
            assert (best_logits[best_loss_index] == logits[best_loss_index].cpu()).all().item()

            acc     = (best_logits.argmax(dim=1) == y.cpu())
            success = (~acc)

            self.do_verbose(
                iteration=iteration,
                loss=loss_indiv.cpu().sum(),
                best_loss=best_loss_indiv.cpu().sum(),
                success=f"{success.sum().item()}/{bs}"
            )
    
    def output(self):
        pass
    



