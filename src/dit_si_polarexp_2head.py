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
    parser.add_argument("--log_freq", type=int, default=200, help="Logging frequency")
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
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon requires PyTorch >= 2.9.")

    muon_params = [
        m.weight
        for block in model.blocks
        for m in (*block.attn.modules(), *block.mlp.modules())
        if isinstance(m, torch.nn.Linear)
    ]
    muon_ids = {id(p) for p in muon_params}
    adamw_params = [p for p in model.parameters() if id(p) not in muon_ids]

    optimizers = [
        torch.optim.Muon(
            muon_params,
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=args.weight_decay,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

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
                f"Total Loss: {loss_total.item():.4f} | "
                f"Loss b: {loss_b.item():.4f} | "
                f"Loss eta: {loss_eta.item():.4f}"
            )

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

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

    # 2. Generate and Log Sampling Comparison Images
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