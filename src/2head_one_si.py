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


class DualHeadVelocityScoreMLP(torch.nn.Module):
    def __init__(self, dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        self.head_b = torch.nn.Linear(hidden_dim, dim)
        self.head_eta = torch.nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, t], dim=-1)
        features = self.backbone(inp)

        b_pred = self.head_b(features)
        eta_pred = self.head_eta(features)
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
        help="Initial (peak) learning rate for the cosine schedule",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Final learning rate reached at the end of the cosine schedule",
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
        help="Hidden dimension of MLP backbone",
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
    model = DualHeadVelocityScoreMLP(dim=ndim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Linear warmup to args.lr, then cosine decay to args.min_lr
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

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    data_dict = {"losses": [], "b_losses": [], "eta_losses": [], "lrs": []}

    print("Starting Training Loop...")
    start_time = time.time()

    for step in range(1, args.num_iterations + 1):
        model.train()
        optimizer.zero_grad()

        x0s = base(args.batch_size)
        x1s = target(args.batch_size, device=device)
        ts = torch.rand(args.batch_size, device=device)

        loss_total, loss_b, loss_eta = compute_loss(
            model, x0s, x1s, ts, interpolant, lambda_b=args.lambda_b
        )

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # LR used for this step, then advance the schedule
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        data_dict["losses"].append(loss_total.item())
        data_dict["b_losses"].append(loss_b.item())
        data_dict["eta_losses"].append(loss_eta.item())
        data_dict["lrs"].append(current_lr)

        # Log metrics to CometML
        experiment.log_metrics(
            {
                "loss_total": loss_total.item(),
                "loss_b": loss_b.item(),
                "loss_eta": loss_eta.item(),
                "lr": current_lr,
            },
            step=step,
        )

        if step % args.log_freq == 0 or step == 1:
            print(
                f"Iter [{step}/{args.num_iterations}] | "
                f"LR: {current_lr:.2e} | "
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