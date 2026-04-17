import torch
import torch.nn as nn


class GrokAlign(nn.Module):
    """
    Joint Jacobian + offset regularization from Theorem 1 of the Normal
    Alignment paper. Adds a single penalty term to the task loss:

        total_loss = task_loss + lambda_reg * (E[||J_x||_F] + E[||b_x||^2_2])

    where J_x is the input-output Jacobian at x, and b_x = f(x) - J_x x is
    the offset of the local affine approximation. This joint constraint is
    exactly what Theorem 1 motivates: ||J_x||^2_F + ||b_x||^2_2 <= α.

    ||J_x||_F is estimated via Hutchinson's estimator (one random output-space
    vector per forward pass). J_x x is estimated via a second forward pass at
    (1+ε)x — exact for CPA (ReLU) networks when ε is small enough to preserve
    all activation patterns.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion,
        lambda_reg: float = 0.1,
        offset_eps: float = 1e-4,
        return_metrics: bool = False,
        device=None,
    ):
        """
        Args:
            model:          The network to regularize.
            criterion:      Loss function taking raw logits and returning a scalar.
            lambda_reg:     Coefficient for the joint Jacobian + offset penalty.
            offset_eps:     Finite-difference step for J_x x estimation (default 1e-4).
            return_metrics: If True, return (total_loss, metrics_dict) instead of scalar.
            device:         Torch device (inferred from model if not given).
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lambda_reg = lambda_reg
        self.offset_eps = offset_eps
        self.return_metrics = return_metrics
        self.device = device if device is not None else next(model.parameters()).device

    def _get_random_projection(self, batch_size, output_shape):
        v = torch.randn(*output_shape, device=self.device)
        v_flat = v.reshape(batch_size, -1)
        norms = v_flat.norm(dim=-1, keepdim=True)
        return (v_flat / torch.clamp(norms, min=1e-8)).view_as(v)

    def compute_jacobian_norm(self, inputs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Estimates E[||J_x||_F] via Hutchinson's estimator with one random
        output-space vector (Algorithm 3 from the paper).
        """
        batch_size = logits.size(0)
        output_dim = logits.numel() // batch_size
        v = self._get_random_projection(batch_size, logits.shape)

        Jv = torch.autograd.grad(
            outputs=logits,
            inputs=inputs,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if Jv is None:
            return torch.tensor(0.0, device=self.device)

        Jv_norm_sq = (Jv.flatten(start_dim=1) ** 2).sum(dim=1)
        return torch.sqrt(torch.clamp(Jv_norm_sq * output_dim, min=1e-8)).mean()

    def compute_offset_norm(self, inputs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes E[||b_x||^2_2] where b_x = f(x) - J_x x.

        J_x x is estimated via a second forward pass:
            J_x x = (f((1+ε)x) - f(x)) / ε
        which is exact for CPA (ReLU) networks when ε preserves all activation
        patterns. Gradients flow through both passes.
        """
        # If a forward-mode JVP was computed earlier and attached to the inputs
        # (as an attribute), use it directly to avoid a second forward pass.
        precomputed_jx = getattr(inputs, "__grokalign_jx", None)
        if precomputed_jx is not None:
            Jx_x = precomputed_jx
        else:
            logits_scaled = self.model(inputs * (1.0 + self.offset_eps))
            Jx_x = (logits_scaled - logits) / self.offset_eps
        b_x = logits - Jx_x
        # Clean up any temporary attribute to avoid surprising statefulness.
        if hasattr(inputs, "__grokalign_jx"):
            try:
                delattr(inputs, "__grokalign_jx")
            except Exception:
                pass
        return (b_x ** 2).sum(dim=-1).mean()

    def forward(self, inputs: torch.Tensor):
        if not inputs.requires_grad:
            inputs.requires_grad_(True)
        # Try to compute logits and J_x x in one forward-mode jvp call.
        # This avoids the second forward pass used for finite-difference J_x x.
        Jx = None
        logits = None
        try:
            from torch.autograd.functional import jvp

            outputs, jvp_out = jvp(self.model, (inputs,), (inputs,), create_graph=True)
            # jvp returns tuples corresponding to the primals; unpack if needed
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            if isinstance(jvp_out, tuple):
                Jx = jvp_out[0]
            else:
                Jx = jvp_out
            # stash Jx on the inputs tensor so compute_offset_norm can reuse it
            try:
                setattr(inputs, "__grokalign_jx", Jx)
            except Exception:
                pass
        except Exception:
            # Fall back to standard single forward pass; offset will use the
            # finite-difference second pass inside compute_offset_norm.
            logits = self.model(inputs)
        task_loss = self.criterion(logits)

        jac_norm = self.compute_jacobian_norm(inputs, logits)
        offset_norm = self.compute_offset_norm(inputs, logits)

        total_loss = task_loss + self.lambda_reg * (jac_norm + offset_norm)

        if self.return_metrics:
            return total_loss, {
                "task_loss": task_loss.item(),
                "jac_norm": jac_norm.item(),
                "offset_norm": offset_norm.item(),
            }
        return total_loss
