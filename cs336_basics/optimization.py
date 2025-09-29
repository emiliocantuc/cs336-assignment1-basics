import torch
from torch.nn import Parameter
from torch import Tensor
from einops import rearrange, einsum, repeat
from jaxtyping import Float, Int, Bool
import math


def cross_entropy(input: Float[Tensor, "... classes"], target: Int[Tensor, "... 1"]):
    if target.dim() == input.dim() - 1:
        target = target.unsqueeze(-1)

    m = input.amax(-1, keepdim=True)
    norm = m + torch.log(torch.exp(input - m).sum(-1, keepdim=True))
    log_probs = input - norm
    loss = -log_probs.gather(-1, target).squeeze(-1)  # assumes one-hot target
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: list[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        assert lr > 0
        assert len(betas) == 2 and all([0 < b < 1 for b in betas])
        super().__init__(params, defaults={"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr, (beta1, beta2), eps, weight_decay = group["lr"], group["betas"], group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data
                state = self.state[p]

                # update moments
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["m"] = beta1 * state["m"] + (1 - beta1) * g
                state["v"] = beta2 * state["v"] + (1 - beta2) * g.pow(2)

                # lr
                t = state.get("t", 1)
                state["t"] = t + 1

                _lr = lr * math.sqrt(1 - (beta2**t)) / (1 - (beta1**t))

                # update
                p.data -= _lr * state["m"] / (state["v"].sqrt() + eps)
                p.data *= 1 - lr * weight_decay


@torch.no_grad()
def gradient_clipping(params: list[Parameter], max_l2_norm: float, eps: float = 1e-6):
    # compute the global norm
    norm = 0
    for p in params:
        if p.grad is not None:
            norm += p.grad.pow(2).sum()

    norm = norm.sqrt()

    if norm > max_l2_norm:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(max_l2_norm / (norm + eps))


def cos_annealing_lr_schedule(t: float, min_lr: float, max_lr: float, warmup_iters: int, cosine_cycle_iters: int):
    if t < warmup_iters:
        return (t / warmup_iters) * max_lr
    elif t > cosine_cycle_iters:
        return min_lr
    else:
        return min_lr + (max_lr - min_lr) * 0.5 * (
            1 + math.cos(math.pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        )
