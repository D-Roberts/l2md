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
# Network Architecture (Diffusion Mamba / DiM)
# ==========================================


class TimestepEmbedder(torch.nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""

    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_size, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class BidirectionalSelectiveSSM(torch.nn.Module):
    """Pure PyTorch Bidirectional Selective State Space Model (BiSSM) Layer."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model

        self.in_proj = torch.nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = torch.nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
        )
        self.x_proj = torch.nn.Linear(
            self.d_inner, self.d_state * 2 + 1, bias=False
        )
        self.dt_proj = torch.nn.Linear(1, self.d_inner, bias=True)

        # Initialize S4D diagonal parameter A and feedthrough parameter D
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
            self.d_inner, 1
        )
        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = torch.nn.Linear(self.d_inner, d_model, bias=False)

    def selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        """Selective scan algorithm across sequence length L."""
        B, L, D = u.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Discretization parameter projections
        x_dbl = self.x_proj(u)  # (B, L, 2*d_state + 1)
        dt_raw, B_ssm, C_ssm = torch.split(
            x_dbl, [1, self.d_state, self.d_state], dim=-1
        )
        dt = torch.nn.functional.softplus(
            self.dt_proj(dt_raw)
        )  # (B, L, d_inner)

        y_list = []
        x_state = torch.zeros(
            B, D, self.d_state, device=u.device, dtype=u.dtype
        )

        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, D, 1)
            u_t = u[:, t, :].unsqueeze(-1)  # (B, D, 1)
            B_t = B_ssm[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            C_t = C_ssm[:, t, :].unsqueeze(1)  # (B, 1, d_state)

            dA = torch.exp(dt_t * A.unsqueeze(0))  # (B, D, d_state)
            dB = dt_t * B_t  # (B, D, d_state)

            x_state = dA * x_state + dB * u_t  # (B, D, d_state)
            y_t = (x_state * C_t).sum(dim=-1)  # (B, D)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)  # (B, L, D)
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xz = self.in_proj(x)
        x_val, z = xz.chunk(2, dim=-1)

        # 1D Convolution over sequence dimension
        x_conv = self.conv1d(x_val.transpose(1, 2)).transpose(1, 2)
        x_act = torch.nn.functional.silu(x_conv)

        # Bidirectional SSM Scanning (Forward + Backward)
        y_fwd = self.selective_scan(x_act)
        y_bwd = torch.flip(
            self.selective_scan(torch.flip(x_act, dims=[1])), dims=[1]
        )
        y_ssm = 0.5 * (y_fwd + y_bwd)

        # Gated multiplicative activation
        y_gated = y_ssm * torch.nn.functional.silu(z)
        return self.out_proj(y_gated)


class DiMBlock(torch.nn.Module):
    """Diffusion Mamba (DiM) Block with Adaptive LayerNorm (AdaLN-Zero)."""

    def __init__(
        self, hidden_dim: int, d_state: int = 16, mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.mamba = BidirectionalSelectiveSSM(hidden_dim, d_state=d_state)
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
        torch.nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        torch.nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_ssm, scale_ssm, gate_ssm, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Mamba SSM Branch
        norm_x = self.norm1(x) * (1 + scale_ssm) + shift_ssm
        ssm_out = self.mamba(norm_x)
        x = x + gate_ssm * ssm_out

        # Feed-Forward MLP Branch
        norm_x2 = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp * mlp_out

        return x


class DualHeadVelocityScoreDiM(torch.nn.Module):
    """Dual-Head Diffusion Mamba backbone replacing MLP/DiT."""

    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 4,
        d_state: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.x_embedder = torch.nn.Linear(dim, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)

        self.blocks = torch.nn.ModuleList(
            [DiMBlock(hidden_dim, d_state=d_state) for _ in range(depth)]
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

        self.head_b = torch.nn.Linear(hidden_dim, dim)
        self.head_eta = torch.nn.Linear(hidden_dim, dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_vector = x.ndim == 2
        if is_vector:
            x = x.unsqueeze(1)  # (B, 1, dim)

        c = self.t_embedder(t).unsqueeze(1)  # (B, 1, hidden_dim)
        h = self.x_embedder(x)  # (B, 1, hidden_dim)

        for block in self.blocks:
            h = block(h, c)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale) + shift

        b_pred = self.head_b(h)
        eta_pred = self.head_eta(h)

        if is_vector:
            b_pred = b_pred.squeeze(1)
            eta_pred = eta_pred.squeeze(1)

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
        description="Dual-Head Stochastic Interpolants Training with DiM"
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
        help="Hidden dimension of DiM backbone",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of DiM blocks",
    )
    parser.add_argument(
        "--d_state",
        type=int,
        default=16,
        help="SSM state dimension",
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

    # Initialize Interpolant and Diffusion Mamba Model
    interpolant = Interpolant(path="one-sided-linear", gamma_type="zero").to(device)
    model = DualHeadVelocityScoreDiM(
        dim=ndim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        d_state=args.d_state,
    ).to(device)

    # Muon for 2D weight matrices inside DiM blocks (MLP and Mamba projections);
    # AdamW for embeddings, 1D SSM params, AdaLN projections, output heads, and biases.
    muon_params = []
    for block in model.blocks:
        # Collect linear weights inside MLP
        for m in block.mlp.modules():
            if isinstance(m, torch.nn.Linear):
                muon_params.append(m.weight)
        # Collect 2D projection weights inside Mamba
        for name, param in block.mamba.named_parameters():
            if param.ndim == 2 and "proj" in name:
                muon_params.append(param)

    # Deduplicate weight tensors
    muon_params = list({id(p): p for p in muon_params}.values())
    muon_ids = {id(p) for p in muon_params}
    adamw_params = [p for p in model.parameters() if id(p) not in muon_ids]

    if hasattr(torch.optim, "Muon"):
        optimizers = [
            torch.optim.Muon(
                muon_params,
                lr=args.muon_lr,
                momentum=args.muon_momentum,
                weight_decay=args.weight_decay,
            ),
            torch.optim.AdamW(
                adamw_params, lr=args.lr, weight_decay=args.weight_decay
            ),
        ]
    else:
        print("torch.optim.Muon unavailable; falling back to AdamW for all parameters.")
        optimizers = [
            torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
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

        # Track LRs for this step, then advance the schedules
        lr_muon = schedulers[0].get_last_lr()[0]
        lr_adamw = schedulers[-1].get_last_lr()[0]
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