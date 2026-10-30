# reference codes: https://github.com/malbergo/stochastic-interpolants

import argparse
import math
import os
import sys
import time
from typing import Tuple

from comet_ml import Experiment
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdiffeq import odeint

# ==========================================
# Interpolant & Distribution Definitions
# ==========================================


def make_gamma(gamma_type="brownian", aval=None):
    if gamma_type == "brownian":
        gamma = lambda t: torch.sqrt(t * (1 - t))
        gamma_dot = lambda t: (1 / (2 * torch.sqrt(t * (1 - t)))) * (1 - 2 * t)
        gg_dot = lambda t: (1 / 2) * (1 - 2 * t)
    elif gamma_type == "zero" or gamma_type is None:
        gamma = gamma_dot = gg_dot = lambda t: torch.zeros_like(t)
    else:
        raise NotImplementedError(f"Gamma type {gamma_type} not implemented.")
    return gamma, gamma_dot, gg_dot


def make_It(path="one-sided-linear", gamma=None, gamma_dot=None, gg_dot=None):
    if path == "one-sided-linear":
        a = lambda t: (1 - t)
        adot = lambda t: -1.0 + 0 * t
        b = lambda t: t
        bdot = lambda t: 1.0 + 0 * t
        It = lambda t, x0, x1: a(t) * x0 + b(t) * x1
        dtIt = lambda t, x0, x1: adot(t) * x0 + bdot(t) * x1
    elif path == "one-sided-trig":
        a = lambda t: torch.cos(0.5 * math.pi * t)
        adot = lambda t: -0.5 * math.pi * torch.sin(0.5 * math.pi * t)
        b = lambda t: torch.sin(0.5 * math.pi * t)
        bdot = lambda t: 0.5 * math.pi * torch.cos(0.5 * math.pi * t)
        It = lambda t, x0, x1: a(t) * x0 + b(t) * x1
        dtIt = lambda t, x0, x1: adot(t) * x0 + bdot(t) * x1
    else:
        raise NotImplementedError(f"Path {path} not implemented.")
    return It, dtIt, (a, adot, b, bdot)


class Interpolant(torch.nn.Module):
    def __init__(self, path: str = "one-sided-linear", gamma_type: str = "zero"):
        super().__init__()
        self.path = path
        self.gamma, self.gamma_dot, self.gg_dot = make_gamma(gamma_type=gamma_type)
        self.It, self.dtIt, ab = make_It(path, self.gamma, self.gamma_dot, self.gg_dot)
        self.a, self.adot, self.b, self.bdot = ab

    def calc_xt(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
        z = torch.randn_like(x0)
        xt = self.It(t, x0, x1) + self.gamma(t) * z
        return xt, z


# ==========================================
# Network Architecture & Distributions
# ==========================================


class TimestepEmbedder(torch.nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""

    def __init__(
        self,
        hidden_dim: int,
        frequency_embedding_size: int = 256,
        t_scale: float = 1000.0,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_size, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # The sinusoidal frequencies (max_period=10000) are designed for integer
        # timesteps in [0, 1000]; with t in [0, 1] almost every feature would be
        # constant, so t is rescaled before embedding.
        self.t_scale = t_scale

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        if t.dim() > 1:
            t = t.squeeze(-1)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 != 0:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(
            t * self.t_scale, self.frequency_embedding_size
        )
        return self.mlp(t_freq)


class Attention(torch.nn.Module):
    """Multi-head self-attention with separate Q, K, V projections.

    Separate matrices (instead of nn.MultiheadAttention's fused in_proj_weight)
    let Muon orthogonalize Q, K and V individually, each with its own shape
    scaling.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v = torch.nn.Linear(hidden_dim, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        head_dim = D // self.num_heads
        q, k, v = (
            lin(x).view(B, N, self.num_heads, head_dim).transpose(1, 2)
            for lin in (self.q, self.k, self.v)
        )  # each (B, heads, N, head_dim)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class DiTBlock(torch.nn.Module):
    """Diffusion Transformer block with Adaptive LayerNorm (AdaLN-Zero) modulation."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.attn = Attention(hidden_dim, num_heads=num_heads)
        self.norm2 = torch.nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(mlp_hidden_dim, hidden_dim),
        )

        # AdaLN-Zero produces 6 modulation values: (scale, shift, gate) x 2
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        # Zero-initialize the modulation projection so the block starts as identity
        torch.nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        torch.nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Multi-Head Attention Branch
        norm_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out = self.attn(norm_x)
        x = x + gate_msa * attn_out

        # Feed-Forward MLP Branch
        norm_x2 = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp * mlp_out

        return x


class DualHeadVelocityScoreDiT(torch.nn.Module):
    """Dual-Head Diffusion Transformer replacing the MLP backbone.

    Each coordinate of x is one token (sequence length = dim), so attention
    mixes information across coordinates. With a single token, softmax over
    one key is identically 1 and attention reduces to a linear map.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.dim = dim
        # Shared scalar-to-vector embedding for each coordinate value, plus a
        # learned position embedding so the tokens know which coordinate they
        # are. The position embedding starts at the same scale as the value
        # embedding so the tokens are distinguishable from the first step.
        self.x_embedder = torch.nn.Linear(1, hidden_dim)
        self.pos_embed = torch.nn.Parameter(torch.randn(1, dim, hidden_dim))
        self.t_embedder = TimestepEmbedder(hidden_dim)

        self.blocks = torch.nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads=num_heads) for _ in range(depth)]
        )

        self.final_norm = torch.nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.final_adaLN = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        torch.nn.init.constant_(self.final_adaLN[1].weight, 0)
        torch.nn.init.constant_(self.final_adaLN[1].bias, 0)

        # Per-token heads: each coordinate token predicts its own component
        self.head_b = torch.nn.Linear(hidden_dim, 1)
        self.head_eta = torch.nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, dim), t: (B,) or (B, 1)
        c = self.t_embedder(t).unsqueeze(1)  # (B, 1, hidden_dim)
        h = self.x_embedder(x.unsqueeze(-1)) + self.pos_embed  # (B, dim, hidden_dim)

        for block in self.blocks:
            h = block(h, c)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale) + shift

        b_pred = self.head_b(h).squeeze(-1)  # (B, dim)
        eta_pred = self.head_eta(h).squeeze(-1)  # (B, dim)
        return b_pred, eta_pred


class SimpleNormal:
    def __init__(self, dim: int, device: str):
        self.dim = dim
        self.device = device

    def __call__(self, bs: int):
        return torch.randn(bs, self.dim, device=self.device)

    def log_prob(self, x: torch.Tensor):
        return -0.5 * (x**2 + math.log(2 * math.pi)).sum(dim=-1)


def target(bs: int, device: str):
    x1 = torch.rand(bs, device=device) * 4 - 2
    x2_ = torch.rand(bs, device=device) - torch.randint(2, (bs,), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)
    return torch.cat([x1[:, None], x2[:, None]], dim=1) * 2


# ==========================================
# Muon Optimizer (Polar Express / Newton-Schulz)
# ==========================================

# Original Muon quintic (Keller Jordan): same coefficients at every step.
JORDAN_COEFFS = [(3.4445, -4.7750, 2.0315)]

# Polar Express (Amsel et al., 2025, arXiv:2505.16932): a different minimax-
# optimal quintic at each step; steps beyond the list reuse the last tuple.
_POLAR_EXPRESS_RAW = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
# Safety factor: p(x) -> p(x / 1.01) for all but the last polynomial, so
# round-off that pushes singular values slightly above 1 cannot blow up.
POLAR_EXPRESS_COEFFS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _POLAR_EXPRESS_RAW[:-1]
] + [_POLAR_EXPRESS_RAW[-1]]


def orthogonalize(
    G: torch.Tensor, steps: int = 5, method: str = "polar_express", eps: float = 1e-7
) -> torch.Tensor:
    """Approximate the polar factor U V^T of a 2D matrix G = U S V^T."""
    if method == "polar_express":
        coeffs, norm_scale = POLAR_EXPRESS_COEFFS, 1.01
    elif method == "jordan":
        coeffs, norm_scale = JORDAN_COEFFS, 1.0
    else:
        raise ValueError(f"Unknown ns_method: {method}")
    coeffs = coeffs[:steps] + [coeffs[-1]] * max(0, steps - len(coeffs))

    # bfloat16 is fast on GPU; on CPU float32 is both faster and more accurate
    X = G.bfloat16() if G.is_cuda else G.float()
    transpose = G.size(0) > G.size(1)
    if transpose:  # iterate on the smaller Gram matrix
        X = X.mT
    # eps keeps an all-zero gradient (adaLN-Zero gates are 0 at init) at zero
    X = X / (X.norm() * norm_scale + eps)
    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Single-device Muon for 2D hidden weight matrices.

    Nesterov momentum, then orthogonalize the update, then scale it by
    sqrt(max(1, rows / cols)) as in the original Muon (and torch.optim.Muon's
    default). With ns_method="jordan" this matches torch.optim.Muon up to
    numerics; "polar_express" swaps in the Polar Express coefficients.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        ns_method: str = "polar_express",
    ):
        if ns_method not in ("polar_express", "jordan"):
            raise ValueError(f"Unknown ns_method: {ns_method}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            ns_method=ns_method,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError("Muon only supports 2D parameters.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - mu)
                update = g.lerp(buf, mu) if group["nesterov"] else buf

                update = orthogonalize(
                    update, steps=group["ns_steps"], method=group["ns_method"]
                )
                update = update * max(1.0, p.size(0) / p.size(1)) ** 0.5

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)

        return loss


# ==========================================
# Loss Calculation & Samplers
# ==========================================


def compute_loss(
    model, x0s, x1s, ts, interpolant, lambda_b: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ts_vec = ts.unsqueeze(-1)
    xt, z = interpolant.calc_xt(ts_vec, x0s, x1s)
    dtIt = interpolant.dtIt(ts_vec, x0s, x1s)

    b_pred, eta_pred = model(xt, ts_vec)

    loss_b = torch.mean((b_pred - dtIt) ** 2)
    loss_eta = torch.mean((eta_pred + x0s) ** 2)
    loss_total = lambda_b * loss_b + loss_eta

    return loss_total, loss_b, loss_eta


def grab(var):
    return var.detach().cpu().numpy()


class ODEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        t_vec = t.expand(x.shape[0], 1)
        b_pred, _ = self.model(x, t_vec)
        return b_pred


@torch.no_grad()
def sample_ode(model, x0, steps=100):
    model.eval()
    t_span = torch.linspace(0.0, 1.0, steps, device=x0.device)
    ode_func = ODEWrapper(model)
    trajectory = odeint(ode_func, x0, t_span, method="dopri5", atol=1e-5, rtol=1e-5)
    return trajectory[-1]


@torch.no_grad()
def sample_sde(model, x0, steps=100, eps=0.1):
    model.eval()
    dt = 1.0 / steps
    x = x0.clone()

    for i in range(steps):
        t_val = i * dt
        t_vec = torch.full((x.shape[0], 1), t_val, device=x.device, dtype=x.dtype)

        b_pred, eta_pred = model(x, t_vec)
        drift = b_pred + eps * eta_pred
        noise = torch.randn_like(x) * math.sqrt(2 * eps * (1 - t_val) * dt)
        x = x + drift * dt + noise

    return x


# ==========================================
# Gradient Tracking (CometML)
# ==========================================


def grad_group_name(name: str) -> str:
    """Map a parameter name to a coarse layer group for gradient tracking."""
    parts = name.split(".")
    if parts[0] == "blocks":
        block, sub = parts[1], parts[2]
        if sub == "attn":
            # Q and K together: their gradients show whether attention is
            # actually mixing tokens (they are exactly zero with a single token)
            label = {
                "q": "attn_qk",
                "k": "attn_qk",
                "v": "attn_v",
                "proj": "attn_out",
            }.get(parts[3], f"attn_{parts[3]}")
        else:
            label = {"mlp": "mlp", "adaLN_modulation": "adaLN"}.get(sub, sub)
        return f"block{block}/{label}"
    return {
        "x_embedder": "embed/x",
        "pos_embed": "embed/x",
        "t_embedder": "embed/t",
        "final_adaLN": "final/adaLN",
        "head_b": "heads",
        "head_eta": "heads",
    }.get(parts[0], parts[0])


class GradientTracker:
    """Gradient statistics logged to CometML, plus history for end-of-run plots.

    Every step: pre-clip total gradient norm, gradient norm per optimizer
    group (Muon / AdamW) and whether clipping fired.
    Every `log_freq` steps: per-layer-group gradient norms and the
    update-to-weight ratio ||W_new - W_old|| / ||W_old|| per optimizer.
    Every `hist_freq` steps: gradient histograms per layer group (Comet 3D
    histograms; large groups are randomly subsampled).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        opt_groups: dict,
        experiment,
        clip_value: float,
        log_freq: int = 100,
        hist_freq: int = 500,
        hist_max_values: int = 100_000,
    ):
        self.experiment = experiment
        self.clip_value = clip_value
        self.log_freq = log_freq
        self.hist_freq = hist_freq
        self.hist_max_values = hist_max_values
        self.opt_groups = opt_groups

        self.layer_params = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.layer_params.setdefault(grad_group_name(name), []).append(p)

        # History for plots
        self.steps, self.total_norms, self.clipped = [], [], []
        self.opt_norms = {k: [] for k in opt_groups}
        self.detail_steps = []
        self.layer_norms = {g: [] for g in self.layer_params}
        self.update_ratios = {k: [] for k in opt_groups}
        self._snapshot = None

    def _detailed(self, step: int) -> bool:
        return step == 1 or step % self.log_freq == 0

    def _hist(self, step: int) -> bool:
        return self.hist_freq > 0 and (step == 1 or step % self.hist_freq == 0)

    @staticmethod
    def _sq_sum(tensors) -> torch.Tensor:
        sq = [t.detach().float().pow(2).sum() for t in tensors]
        return torch.stack(sq).sum() if sq else torch.zeros(())

    def _grad_norm(self, params) -> float:
        grads = [p.grad for p in params if p.grad is not None]
        return self._sq_sum(grads).sqrt().item()

    @torch.no_grad()
    def before_clip(self, step: int):
        """Call after backward() and before clipping: records raw gradients."""
        metrics = {}
        for k, params in self.opt_groups.items():
            v = self._grad_norm(params)
            self.opt_norms[k].append(v)
            metrics[f"grad_norm/{k}"] = v

        if self._detailed(step):
            self.detail_steps.append(step)
            for g, params in self.layer_params.items():
                v = self._grad_norm(params)
                self.layer_norms[g].append(v)
                metrics[f"grad_norm_layer/{g}"] = v

        if self._hist(step):
            for g, params in self.layer_params.items():
                grads = [p.grad.flatten() for p in params if p.grad is not None]
                if not grads:
                    continue
                vals = torch.cat(grads).float()
                if vals.numel() > self.hist_max_values:
                    idx = torch.randint(
                        vals.numel(), (self.hist_max_values,), device=vals.device
                    )
                    vals = vals[idx]
                self.experiment.log_histogram_3d(
                    vals.cpu().numpy(), name=f"grad_{g.replace('/', '_')}", step=step
                )

        self.experiment.log_metrics(metrics, step=step)

    @torch.no_grad()
    def after_clip(self, step: int, total_norm: float):
        """Call with the pre-clip total norm returned by clip_grad_norm_."""
        clipped = total_norm > self.clip_value
        self.steps.append(step)
        self.total_norms.append(total_norm)
        self.clipped.append(clipped)
        self.experiment.log_metrics(
            {"grad_norm/total_preclip": total_norm, "grad_clipped": float(clipped)},
            step=step,
        )
        if self._detailed(step):
            self._snapshot = {
                k: [p.detach().clone() for p in params]
                for k, params in self.opt_groups.items()
            }

    @torch.no_grad()
    def after_step(self, step: int):
        """Call after the optimizer steps: update-to-weight ratio per optimizer."""
        if self._snapshot is None:
            return
        metrics = {}
        for k, params in self.opt_groups.items():
            old = self._snapshot[k]
            num = self._sq_sum([p - p0 for p, p0 in zip(params, old)]).sqrt()
            den = self._sq_sum(old).sqrt()
            r = (num / (den + 1e-12)).item()
            self.update_ratios[k].append(r)
            metrics[f"update_ratio/{k}"] = r
        self.experiment.log_metrics(metrics, step=step)
        self._snapshot = None


# ==========================================
# Gradient Visualisation
# ==========================================

# Chart chrome and categorical slots 1-2 (blue, orange) from a validated palette
_INK, _INK_2, _MUTED = "#0b0b0b", "#52514e", "#898781"
_GRID, _AXIS, _SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
_SERIES = {"total": "#2a78d6", "muon": "#2a78d6", "adamw": "#eb6834"}
_SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style_axes(ax):
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_INK_2, labelsize=9)
    ax.xaxis.label.set_color(_INK_2)
    ax.yaxis.label.set_color(_INK_2)
    ax.title.set_color(_INK)


def _rolling_mean(y, window: int) -> np.ndarray:
    """Trailing mean with the same length as y (shorter window at the start)."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) == 0:
        return y
    c = np.concatenate([[0.0], np.cumsum(y)])
    idx = np.arange(1, len(y) + 1)
    start = np.maximum(0, idx - window)
    return (c[idx] - c[start]) / (idx - start)


def _positive(y) -> np.ndarray:
    """Mask non-positive values (e.g. zero Muon gradients at step 1) for log axes."""
    y = np.asarray(y, dtype=float)
    return np.where(y > 0, y, np.nan)


def plot_gradient_norms(tracker: GradientTracker):
    steps = np.asarray(tracker.steps)
    window = max(1, len(steps) // 50)
    frac = 100 * float(np.mean(tracker.clipped)) if tracker.clipped else 0.0

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.patch.set_facecolor(_SURFACE)

    ax_top.plot(steps, _positive(tracker.total_norms), color=_SERIES["total"],
                alpha=0.25, linewidth=0.8, label="per step")
    ax_top.plot(steps, _positive(_rolling_mean(tracker.total_norms, window)),
                color=_SERIES["total"], linewidth=1.6, label=f"rolling mean ({window} steps)")
    ax_top.axhline(tracker.clip_value, color=_MUTED, linewidth=1.0, linestyle="--")
    ax_top.annotate(f"clip threshold ({tracker.clip_value:g})",
                    xy=(1.0, tracker.clip_value), xycoords=("axes fraction", "data"),
                    xytext=(-4, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=9, color=_INK_2)
    ax_top.set_yscale("log")
    ax_top.set_ylabel("gradient norm")
    ax_top.set_title(f"Total gradient norm before clipping (clipped on {frac:.1f}% of steps)",
                     loc="left", fontsize=11)
    ax_top.legend(frameon=False, fontsize=9, labelcolor=_INK_2, loc="upper right")
    _style_axes(ax_top)

    for k, norms in tracker.opt_norms.items():
        smooth = _positive(_rolling_mean(norms, window))
        ax_bot.plot(steps, smooth, color=_SERIES.get(k, _INK_2), linewidth=1.6, label=k)
        last = np.flatnonzero(np.isfinite(smooth))
        if last.size:  # direct label at the line end
            i = last[-1]
            ax_bot.annotate(k, xy=(steps[i], smooth[i]), xytext=(4, 0),
                            textcoords="offset points", va="center",
                            fontsize=9, color=_INK_2)
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("iteration")
    ax_bot.set_ylabel("gradient norm")
    ax_bot.set_title(f"Gradient norm by optimizer group (rolling mean, {window} steps)",
                     loc="left", fontsize=11)
    ax_bot.legend(frameon=False, fontsize=9, labelcolor=_INK_2, loc="upper right")
    _style_axes(ax_bot)

    fig.tight_layout()
    return fig


def plot_update_ratios(tracker: GradientTracker):
    steps = np.asarray(tracker.detail_steps)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(_SURFACE)
    for k, ratios in tracker.update_ratios.items():
        ax.plot(steps, _positive(ratios), color=_SERIES.get(k, _INK_2),
                linewidth=1.6, marker="o", markersize=4, label=k)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("‖ΔW‖ / ‖W‖")
    ax.set_title("Update-to-weight ratio per optimizer step", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_2, loc="upper right")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_layer_grad_heatmap(tracker: GradientTracker):
    from matplotlib.colors import LinearSegmentedColormap

    groups = list(tracker.layer_norms)
    steps = np.asarray(tracker.detail_steps)
    norms = np.array([tracker.layer_norms[g] for g in groups], dtype=float)
    log_norms = np.log10(np.where(norms > 0, norms, np.nan))

    cmap = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE)
    cmap.set_bad(_SURFACE)  # zero gradients (e.g. at step 1) stay blank

    fig, ax = plt.subplots(figsize=(10, 0.28 * len(groups) + 1.6))
    fig.patch.set_facecolor(_SURFACE)
    im = ax.imshow(log_norms, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    n_ticks = min(8, len(steps))
    tick_idx = np.unique(np.linspace(0, len(steps) - 1, n_ticks).round().astype(int))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(steps[i]) for i in tick_idx])
    ax.set_xlabel("iteration")
    ax.set_title("Per-layer gradient norm before clipping (log10)", loc="left", fontsize=11)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=_INK_2, labelsize=8)
    ax.xaxis.label.set_color(_INK_2)
    ax.title.set_color(_INK)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("log10 gradient norm", color=_INK_2, fontsize=9)
    cbar.ax.tick_params(colors=_INK_2, labelsize=8)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    return fig


# ==========================================
# Main Execution Block
# ==========================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dual-Head Stochastic Interpolants Training"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=5000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-3,
        help="Peak AdamW learning rate (input layer, output heads, biases)",
    )
    parser.add_argument(
        "--muon_lr",
        type=float,
        default=0.02,
        help="Peak Muon learning rate (hidden weight matrices)",
    )
    parser.add_argument(
        "--muon_momentum", type=float, default=0.95, help="Muon momentum"
    )
    parser.add_argument(
        "--ns_method",
        type=str,
        default="polar_express",
        choices=["polar_express", "jordan"],
        help="Muon orthogonalization: Polar Express or the original Newton-Schulz",
    )
    parser.add_argument(
        "--ns_steps",
        type=int,
        default=5,
        help="Number of orthogonalization iterations per Muon step",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Decoupled weight decay for both Muon and AdamW",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Final AdamW learning rate; the Muon lr decays by the same ratio",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Linear warmup steps from ~0 up to --lr before cosine decay (0 disables)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of DiT backbone",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of DiT blocks",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads in DiT",
    )
    parser.add_argument(
        "--lambda_b",
        type=float,
        default=0.9,
        help="Velocity loss weighting multiplier",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Max total gradient norm"
    )
    parser.add_argument("--log_freq", type=int, default=200, help="Logging frequency")
    parser.add_argument(
        "--grad_log_freq",
        type=int,
        default=100,
        help="Steps between per-layer gradient norms and update ratios",
    )
    parser.add_argument(
        "--grad_hist_freq",
        type=int,
        default=500,
        help="Steps between gradient histograms in Comet (0 disables)",
    )
    parser.add_argument(
        "--comet_project_name",
        type=str,
        default="l2md",
        help="CometML project name",
    )
    parser.add_argument(
        "--comet_api_key",
        type=str,
        default=None,
        help="CometML API key (optional if set in env as COMET_API_KEY)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize CometML
    experiment = Experiment(
        api_key=args.comet_api_key,
        workspace="l2md",
        project_name=args.comet_project_name,
        auto_output_logging="simple",
    )
    experiment.log_parameters(vars(args))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Torch version: {torch.__version__}")

    ndim = 2
    base = SimpleNormal(ndim, device=device)

    # Initialize Model, Optimizer, Scheduler, Loss Weight, and Interpolant
    interpolant = Interpolant(path="one-sided-linear", gamma_type="zero").to(device)
    model = DualHeadVelocityScoreDiT(
        dim=ndim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    ).to(device)

    # Muon for the 2D weight matrices inside the DiT blocks (Q, K, V, attention
    # output projection, MLP); AdamW for embeddings, position embedding, AdaLN
    # projections, output heads and all biases.
    muon_params = [
        m.weight
        for block in model.blocks
        for m in (*block.attn.modules(), *block.mlp.modules())
        if isinstance(m, torch.nn.Linear)
    ]
    muon_ids = {id(p) for p in muon_params}
    adamw_params = [p for p in model.parameters() if id(p) not in muon_ids]

    optimizers = [
        Muon(
            muon_params,
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=args.weight_decay,
            ns_steps=args.ns_steps,
            ns_method=args.ns_method,
        ),
        torch.optim.AdamW(adamw_params, lr=args.lr, weight_decay=args.weight_decay),
    ]

    # Linear warmup to peak lr, then cosine decay
    if not 0 <= args.warmup_steps < args.num_iterations:
        raise ValueError("--warmup_steps must be in [0, num_iterations).")
    min_ratio = args.min_lr / args.lr
    decay_steps = args.num_iterations - args.warmup_steps

    def lr_lambda(i: int) -> float:
        if i < args.warmup_steps:
            return (i + 1) / args.warmup_steps
        progress = min(1.0, (i - args.warmup_steps) / max(1, decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda) for opt in optimizers
    ]

    tracker = GradientTracker(
        model,
        {"muon": muon_params, "adamw": adamw_params},
        experiment,
        clip_value=args.grad_clip,
        log_freq=args.grad_log_freq,
        hist_freq=args.grad_hist_freq,
    )

    data_dict = {
        "losses": [],
        "b_losses": [],
        "eta_losses": [],
        "lrs_muon": [],
        "lrs_adamw": [],
    }

    print("Starting Training Loop...")
    start_time = time.time()

    for step in range(1, args.num_iterations + 1):
        model.train()
        for opt in optimizers:
            opt.zero_grad()

        x0s = base(args.batch_size)
        x1s = target(args.batch_size, device=device)
        ts = torch.rand(args.batch_size, device=device)

        loss_total, loss_b, loss_eta = compute_loss(
            model, x0s, x1s, ts, interpolant, lambda_b=args.lambda_b
        )

        loss_total.backward()
        tracker.before_clip(step)
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        ).item()
        tracker.after_clip(step, total_norm)
        for opt in optimizers:
            opt.step()
        tracker.after_step(step)

        # LRs used for this step, then advance the schedules
        lr_muon = schedulers[0].get_last_lr()[0]
        lr_adamw = schedulers[1].get_last_lr()[0]
        for sched in schedulers:
            sched.step()

        data_dict["losses"].append(loss_total.item())
        data_dict["b_losses"].append(loss_b.item())
        data_dict["eta_losses"].append(loss_eta.item())
        data_dict["lrs_muon"].append(lr_muon)
        data_dict["lrs_adamw"].append(lr_adamw)

        # Log metrics to CometML
        experiment.log_metrics(
            {
                "loss_total": loss_total.item(),
                "loss_b": loss_b.item(),
                "loss_eta": loss_eta.item(),
                "lr_muon": lr_muon,
                "lr_adamw": lr_adamw,
            },
            step=step,
        )

        if step % args.log_freq == 0 or step == 1:
            print(
                f"Iter [{step}/{args.num_iterations}] | "
                f"LR muon/adamw: {lr_muon:.2e}/{lr_adamw:.2e} | "
                f"Grad norm: {total_norm:.3f} | "
                f"Total Loss: {loss_total.item():.4f} | "
                f"Loss b: {loss_b.item():.4f} | "
                f"Loss eta: {loss_eta.item():.4f}"
            )

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")
    frac_clipped = float(np.mean(tracker.clipped))
    experiment.log_metric("frac_steps_clipped", frac_clipped)
    print(f"Gradient clipping fired on {100 * frac_clipped:.1f}% of steps.")

    # 1. Plot & Log Training Loss Curves
    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(data_dict["losses"], label="Total Loss (b + eta)", alpha=0.7)
    ax_loss.plot(data_dict["b_losses"], label="Velocity Loss (b)", alpha=0.8)
    ax_loss.plot(data_dict["eta_losses"], label="Score/Eta Loss (eta)", alpha=0.8)
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss Value")
    ax_loss.set_title("Training Loss Curves")
    ax_loss.legend()
    ax_loss.grid(True)
    plt.tight_layout()

    # Log figure to CometML
    experiment.log_figure(figure_name="training_loss_curves", figure=fig_loss)
    plt.close(fig_loss)

    # 2. Gradient diagnostics figures
    for fig_name, fig in (
        ("gradient_norms", plot_gradient_norms(tracker)),
        ("update_to_weight_ratio", plot_update_ratios(tracker)),
        ("layer_gradient_heatmap", plot_layer_grad_heatmap(tracker)),
    ):
        experiment.log_figure(figure_name=fig_name, figure=fig)
        plt.close(fig)

    # 3. Generate and Log Sampling Comparison Images
    num_samples = 10000
    x0_test = base(num_samples)

    print("Generating ODE Probability Flow samples...")
    xf_ode = grab(sample_ode(model, x0_test, steps=100))

    print("Generating SDE samples...")
    xf_sde = grab(sample_sde(model, x0_test, steps=100, eps=0.05))

    target_pts = grab(target(num_samples, device=device))

    fig_samples, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(target_pts[:, 0], target_pts[:, 1], s=2, alpha=0.3, color="black")
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-6.5, 6.5)
    axes[0].set_title("Target Distribution")

    axes[1].scatter(xf_ode[:, 0], xf_ode[:, 1], s=2, alpha=0.3, color="blue")
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-6.5, 6.5)
    axes[1].set_title("ODE Probability Flow Samples")

    axes[2].scatter(xf_sde[:, 0], xf_sde[:, 1], s=2, alpha=0.3, color="purple")
    axes[2].set_xlim(-5, 5)
    axes[2].set_ylim(-6.5, 6.5)
    axes[2].set_title("SDE Flow Samples")

    plt.tight_layout()

    # Log sampling comparison figure to CometML
    experiment.log_figure(figure_name="sampling_comparison", figure=fig_samples)
    plt.close(fig_samples)

    experiment.end()
    print("CometML experiment completed successfully.")


if __name__ == "__main__":
    main()