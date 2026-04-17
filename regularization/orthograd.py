import torch


class OrthoGrad:
    """
    Optimizer wrapper that orthogonalizes gradients before each update step.

    For each parameter tensor, removes the gradient component parallel to the
    current weights (Gram-Schmidt projection). This constrains updates to change
    only the direction of weights, not their magnitude.
    """

    def __init__(self, params, base_optimizer_cls, **base_kwargs):
        self.base_optimizer = base_optimizer_cls(params, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def _orthogonalize_gradients(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                w_flat = p.data.flatten()
                g_flat = p.grad.data.flatten()
                w_norm_sq = (w_flat * w_flat).sum()
                if w_norm_sq > 1e-8:
                    proj = (g_flat * w_flat).sum() / w_norm_sq
                    p.grad.data = p.grad.data - proj * p.data

    def step(self, closure=None):
        self._orthogonalize_gradients()
        return self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
