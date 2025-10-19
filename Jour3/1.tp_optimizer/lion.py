from typing import Tuple, Optional, Callable
import torch
from torch.optim.optimizer import Optimizer

# === Try to import Triton ===
has_triton = True
try:
    import triton
except ImportError:
    has_triton = False
    print("[Warning] Triton import failed: module 'triton' not found.")

try:
    import triton.language as tl
except ImportError:
    has_triton = False
    print("[Warning] Triton import failed: module 'triton.language' not found.")

# === Default PyTorch update function ===
def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    p.data.mul_(1 - lr * wd)
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.add_(update, alpha=-lr)
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

# === Triton version, only if available ===
if has_triton:
    @triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ], key=['n_elements'])
    @triton.jit
    def update_fn_kernel(
        p_ptr, grad_ptr, exp_avg_ptr,
        lr, wd, beta1, beta2,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        p = tl.load(p_ptr + offsets, mask=mask)
        grad = tl.load(grad_ptr + offsets, mask=mask)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)

        p = p * (1 - lr * wd)
        diff = exp_avg - grad
        update = diff * beta1 + grad
        can_update = update != 0
        update_sign = tl.where(update > 0, -lr, lr)
        p = p + update_sign * can_update
        exp_avg = diff * beta2 + grad

        tl.store(p_ptr + offsets, p, mask=mask)
        tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)

    def triton_update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
        assert all([t.is_cuda for t in (p, grad, exp_avg)])
        n_elements = p.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        update_fn_kernel[grid](
            p, grad, exp_avg, lr, wd, beta1, beta2, n_elements
        )

# === Utility ===
def exists(val):
    return val is not None

# === Lion Optimizer ===
class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.95, 0.98),
        weight_decay: float = 0.0,
        use_triton: bool = False
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )

        super().__init__(params, defaults)

        if use_triton:
            if has_triton:
                self.update_fn = triton_update_fn
            else:
                print("[Warning] Triton was requested but is not available. Falling back to PyTorch update_fn.")
                self.update_fn = update_fn
        else:
            self.update_fn = update_fn

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
                grad = p.grad
                lr = group['lr']
                wd = group['weight_decay']
                beta1, beta2 = group['betas']
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss
