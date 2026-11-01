"Multitask Hadamard Inpainting in Continuous Space"

import time
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint

if torch.cuda.is_available():
    device = "cuda:0"
    print("Using gpu")
else:
    device = "cpu"
    print("Using cpu.")

print("Torch version:", torch.__version__)
torch.manual_seed(0)
np.random.seed(0)


Time = torch.tensor
Alpha = torch.tensor
Sample = torch.tensor


class HadamardInterpolant(torch.nn.Module):
    def calc_xt(self, alpha: Alpha, x0: Sample, x1: Sample) -> Sample:
        return alpha * x0 + (1 - alpha) * x1

    @staticmethod
    def regression_target(x0: Sample, x1: Sample) -> Sample:
        return x0 - x1

    @staticmethod
    def eta0(eta: Sample, alpha: Alpha, x: Sample) -> Sample:
        return x + (1 - alpha) * eta

    @staticmethod
    def eta1(eta: Sample, alpha: Alpha, x: Sample) -> Sample:
        return x - alpha * eta

    @staticmethod
    def velocity(eta: Sample, alpha_dot: Alpha) -> Sample:
        return alpha_dot * eta


def sample_alpha(
    bs: int, d: int, mix=(0.3, 0.2, 0.5), mask_probs=(0.4, 0.4, 0.2)
) -> Alpha:
    t = torch.rand(bs, 1, device=device)
    kind = torch.multinomial(torch.tensor(mix), bs, replacement=True).to(device)[
        :, None
    ]
    diag = t.expand(bs, d)
    unif = torch.rand(bs, d, device=device)
    cat = (
        torch.multinomial(torch.tensor(mask_probs), bs * d, replacement=True)
        .view(bs, d)
        .to(device)
    )
    mask = torch.where(
        cat == 0,
        torch.zeros_like(unif),
        torch.where(cat == 1, diag, torch.ones_like(unif)),
    )
    return torch.where(kind == 0, diag, torch.where(kind == 1, unif, mask))


def grab(var):
    return var.detach().cpu().numpy()


def in_support(x: Sample) -> float:
    X, Y = x[:, 0], x[:, 1]
    inside = (X >= -4) & (X < 4) & (Y >= -4) & (Y < 4)
    parity = (torch.floor(X / 2) + torch.floor(Y / 2)) % 2 == 0
    return (inside & parity).float().mean().item()


def smooth(y, k=50):
    k = max(1, min(k, len(y)))
    return np.convolve(np.asarray(y, dtype=float), np.ones(k) / k, mode="valid")


ndim = 2


def target(bs):
    x1 = torch.rand(bs) * 4 - 2
    x2_ = torch.rand(bs) - torch.randint(2, (bs,)) * 2
    x2 = x2_ + (torch.floor(x1) % 2)
    return (torch.cat([x1[:, None], x2[:, None]], 1) * 2).to(device)


target_samples = grab(target(10000))


fig = plt.figure(figsize=(6, 6))
plt.hist2d(
    target_samples[:, 0], target_samples[:, 1], bins=100, range=[[-4, 4], [-4, 4]]
)
plt.title("Checker Target")
plt.show()


print("Batch Shape:", target_samples.shape)
print(
    "on-support fraction of target samples (sanity check, should be 1):",
    in_support(target(10000)),
)


class Prior(torch.nn.Module):
    def forward(self, batch_size):
        raise NotImplementedError()

    def log_prob(self, x):
        raise NotImplementedError()

    def draw(self, batch_size):
        return self.forward(batch_size)


class SimpleNormal(Prior):
    def __init__(self, loc, var, requires_grad=False):
        super().__init__()

        if requires_grad:
            loc.requires_grad_()
            var.requires_grad_()
        self.loc = loc
        self.var = var
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(self.loc), torch.flatten(self.var)
        )
        self.shape = loc.shape

    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)

    def forward(self, batch_size):
        x = self.dist.sample((batch_size,))
        return torch.reshape(x, (-1,) + self.shape)

    def rsample(self, batch_size):
        x = self.dist.rsample((batch_size,))
        return torch.reshape(x, (-1,) + self.shape)


base_loc = torch.zeros(ndim, device=device)
base_var = torch.ones(ndim, device=device)
base = SimpleNormal(base_loc, 1.0 * base_var)
base_samples = grab(base(20000))


fig = plt.figure(
    figsize=(
        6,
        6,
    )
)
plt.scatter(base_samples[:, 0], base_samples[:, 1], label="base", alpha=0.2)
plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.2)
plt.title("Base vs Target")
plt.show()


interpolant = HadamardInterpolant()


def loss_fn_eta(
    eta: torch.nn.Module,
    x0s: Sample,
    x1s: Sample,
    alphas: Alpha,
    interpolant: HadamardInterpolant,
) -> torch.tensor:
    xt = interpolant.calc_xt(alphas, x0s, x1s)
    eta_t = eta(xt, alphas)
    target = interpolant.regression_target(x0s, x1s)
    return torch.sum((eta_t - target) ** 2, dim=-1).mean()


class InputWrapper(torch.nn.Module):
    def __init__(self, v):
        super(InputWrapper, self).__init__()
        self.v = v

    def net_inp(self, x: torch.tensor, alpha: torch.tensor) -> torch.tensor:
        return torch.cat((x, alpha.expand_as(x)), dim=1)

    def forward(self, x, alpha):
        return self.v(self.net_inp(x, alpha))


def make_activation(act):
    if act == "silu":
        return torch.nn.SiLU()
    elif act == "none" or act is None:
        return None
    else:
        raise NotImplementedError(f"Unknown activation function {act}")


def make_fc_net(hidden_sizes, in_size, out_size, inner_act, final_act, **config):
    sizes = [in_size] + hidden_sizes + [out_size]
    net = []
    for i in range(len(sizes) - 1):
        net.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i != len(sizes) - 2:
            net.append(make_activation(inner_act))
            continue
        else:
            if make_activation(final_act):
                net.append(make_activation(final_act))

    v_net = torch.nn.Sequential(*net)
    return InputWrapper(v_net)


base_lr = 2e-3
hidden_sizes = [256, 256, 256, 256]
in_size = 2 * ndim
out_size = ndim
inner_act = "silu"
final_act = "none"
print_model = True


eta = make_fc_net(
    hidden_sizes=hidden_sizes,
    in_size=in_size,
    out_size=out_size,
    inner_act=inner_act,
    final_act=final_act,
).to(device)
opt = torch.optim.Adam(eta.parameters(), lr=base_lr)
sched = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=1500, gamma=0.4)


eps_bar = 1.0
N_era = 14
N_epoch = 500
bs = 2000
plot_bs = 5000
likelihood_bs = 1000
metrics_freq = 500
plot_freq = 3500
n_step = 100
n_step_sde = 200


if print_model:
    print("Here's the model eta:", eta)


@dataclass
class MaskPath:
    known: torch.Tensor
    active: torch.Tensor

    def alpha(self, t):
        return self.active * (1 - t) + (1 - self.known - self.active)

    def alpha_dot(self, t):
        return -self.active


def unconditional_path(d):
    return MaskPath(
        known=torch.zeros(d, device=device), active=torch.ones(d, device=device)
    )


def inpainting_path(obs):
    obs = obs.to(device).float()
    return MaskPath(known=obs, active=1 - obs)


def sequential_paths(order, d):
    paths = []
    for k, i in enumerate(order):
        known = torch.zeros(d, device=device)
        active = torch.zeros(d, device=device)
        for j in order[:k]:
            known[j] = 1
        active[i] = 1
        paths.append(MaskPath(known, active))
    return paths


def velocity_field(eta, path, x, t):
    a = path.alpha(t).expand_as(x)
    ad = path.alpha_dot(t).expand_as(x)
    return HadamardInterpolant.velocity(eta(x, a), ad)


@torch.no_grad()
def sample_ode(eta, path, x_init, n_step=100):
    x = x_init.clone()
    ts = torch.linspace(0.0, 1.0, n_step + 1, device=x.device)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        h = t1 - t0
        k1 = velocity_field(eta, path, x, t0)
        k2 = velocity_field(eta, path, x + h * k1, t1)
        x = x + 0.5 * h * (k1 + k2)
    return x


@torch.no_grad()
def sample_sde(eta, path, x_init, n_step=200, eps_bar=1.0):
    x = x_init.clone()
    ts = torch.linspace(0.0, 1.0, n_step + 1, device=x.device)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        h = t1 - t0
        a = path.alpha(t0).expand_as(x)
        ad = path.alpha_dot(t0).expand_as(x)
        on = (ad != 0).to(x)
        e = eta(x, a)
        e0 = HadamardInterpolant.eta0(e, a, x)
        drift = HadamardInterpolant.velocity(e, ad) - eps_bar * on * e0
        x = x + h * drift + on * torch.sqrt(2 * eps_bar * a * h) * torch.randn_like(x)
    return x


def run_stages(sampler, eta, paths, x_init, **kw):
    x = x_init
    for p in paths:
        x = sampler(eta, p, x, **kw)
    return x


def log_likelihood(eta, x0, rtol=1e-4, atol=1e-4):
    path = unconditional_path(x0.shape[1])

    def rhs(t, state):
        x = state[0]
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            v = velocity_field(eta, path, x, t)
            div = sum(
                torch.autograd.grad(v[:, i].sum(), x, retain_graph=True)[0][:, i]
                for i in range(x.shape[1])
            )
        return v.detach(), -div.detach()

    with torch.no_grad():
        xs, dlogp = odeint(
            rhs,
            (x0, torch.zeros(x0.shape[0], device=x0.device)),
            torch.tensor([0.0, 1.0], device=x0.device),
            method="dopri5",
            rtol=rtol,
            atol=atol,
        )
    return xs[-1], base.log_prob(x0) + dlogp[-1]


def log_metrics(eta, likelihood_bs, counter, data_dict):
    _, logpx = log_likelihood(eta, base(likelihood_bs))
    data_dict["logp_steps"].append(counter)
    data_dict["logps_pflow"].append(grab(logpx).mean())


def make_plots(eta, counter, data_dict, plot_bs):
    z, x1 = base(plot_bs), target(plot_bs)
    obs = torch.tensor([1.0, 0.0], device=device)
    x_unc = sample_ode(eta, unconditional_path(ndim), z, n_step=n_step)
    x_inp = sample_ode(
        eta, inpainting_path(obs), obs * x1 + (1 - obs) * z, n_step=n_step
    )
    x_seq = run_stages(
        sample_ode, eta, sequential_paths([0, 1], ndim), z, n_step=n_step
    )

    print("EPOCH:", counter)
    print("LOSS, GRAD:", data_dict["losses"][-1], data_dict["eta_grads"][-1])
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    axes[0].plot(smooth(data_dict["losses"]))
    axes[0].set_title("LOSS (moving avg; floor > 0)")

    axes[1].plot(
        data_dict["logp_steps"], data_dict["logps_pflow"], color="purple", label="pflow"
    )
    axes[1].axhline(-np.log(32), color="k", ls="--", lw=1, label=r"true: $-\log 32$")
    axes[1].set_ylim(-7, 0)
    axes[1].legend(loc="best")
    axes[1].set_title(r"$\log p$ from PFlow (unconditional path)")

    for ax, xx, title in [
        (axes[2], x_unc, "Unconditional (diagonal path)"),
        (axes[3], x_inp, "Inpainting: X observed, Y generated"),
        (axes[4], x_seq, "Sequential: X, then Y | X"),
    ]:
        xn = grab(xx)
        ax.scatter(xn[:, 0], xn[:, 1], s=2, alpha=0.2)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_title(f"{title}\non support: {in_support(xx):.3f}", fontsize=11)
    plt.show()


def train_step(bs: int, interpolant: HadamardInterpolant, opt: Any, sched: Any):
    opt.zero_grad()

    x0s = base(bs)
    x1s = target(bs)
    alphas = sample_alpha(bs, ndim)

    loss_val = loss_fn_eta(eta, x0s, x1s, alphas, interpolant)
    loss_val.backward()
    eta_grad = torch.nn.utils.clip_grad_norm_(eta.parameters(), float("inf"))
    opt.step()
    sched.step()

    return loss_val.detach(), eta_grad.detach()


data_dict = {
    "losses": [],
    "eta_grads": [],
    "logp_steps": [],
    "logps_pflow": [],
}

counter = 1
start = time.perf_counter()
for i, era in enumerate(range(N_era)):
    for j, epoch in enumerate(range(N_epoch)):
        loss, eta_grad = train_step(bs, interpolant, opt, sched)
        data_dict["losses"].append(grab(loss))
        data_dict["eta_grads"].append(grab(eta_grad))

        if (counter - 1) % metrics_freq == 0:
            log_metrics(eta, likelihood_bs, counter, data_dict)

        if (counter - 1) % plot_freq == 0:
            make_plots(eta, counter, data_dict, plot_bs)

        counter += 1

print(f"trained {counter - 1} steps in {time.perf_counter() - start:.0f}s")
log_metrics(eta, likelihood_bs, counter - 1, data_dict)
make_plots(eta, counter - 1, data_dict, plot_bs)


torch.manual_seed(1)
n = 5000
z, x1 = base(n), target(n)
obsX = torch.tensor([1.0, 0.0], device=device)
obsY = torch.tensor([0.0, 1.0], device=device)
obsR = torch.nn.functional.one_hot(torch.randint(2, (n,)), 2).float().to(device)

tasks = {
    "unconditional": lambda smp: smp(eta, unconditional_path(ndim), z),
    "inpaint: X observed": lambda smp: smp(
        eta, inpainting_path(obsX), obsX * x1 + (1 - obsX) * z
    ),
    "inpaint: Y observed": lambda smp: smp(
        eta, inpainting_path(obsY), obsY * x1 + (1 - obsY) * z
    ),
    "inpaint: random coord": lambda smp: smp(
        eta, inpainting_path(obsR), obsR * x1 + (1 - obsR) * z
    ),
    "sequential: X then Y": lambda smp: run_stages(
        smp, eta, sequential_paths([0, 1], ndim), z
    ),
    "sequential: Y then X": lambda smp: run_stages(
        smp, eta, sequential_paths([1, 0], ndim), z
    ),
}
ode = partial(sample_ode, n_step=n_step)
sde = partial(sample_sde, n_step=n_step_sde, eps_bar=eps_bar)
samples = {name: (fn(ode), fn(sde)) for name, fn in tasks.items()}

print(
    f"{'task':24s} {'ODE':>6s} {'SDE':>6s}    (on-support fraction; target = 1, noise ~ {in_support(z):.2f})"
)
for name, (xo, xs) in samples.items():
    print(f"{name:24s} {in_support(xo):6.3f} {in_support(xs):6.3f}")

for name, obs in [
    ("inpaint: X observed", obsX),
    ("inpaint: Y observed", obsY),
    ("inpaint: random coord", obsR),
]:
    xo, xs = samples[name]
    o = obs.expand_as(x1).bool()
    print(
        f"max change of observed coords, {name:22s}: ODE {(xo - x1)[o].abs().max().item():.1e}, "
        f"SDE {(xs - x1)[o].abs().max().item():.1e}"
    )

fig, axes = plt.subplots(2, 7, figsize=(28, 8))
for r, (xx, lab) in enumerate([(x1, "target"), (z, "base noise")]):
    xn = grab(xx)
    axes[r, 0].scatter(
        xn[:, 0], xn[:, 1], s=1, alpha=0.2, color="C1" if r == 0 else "C0"
    )
    axes[r, 0].set_title(lab)
for c, (name, (xo, xs)) in enumerate(samples.items(), start=1):
    for r, (xx, lab) in enumerate([(xo, "ODE"), (xs, "SDE")]):
        xn = grab(xx)
        axes[r, c].scatter(xn[:, 0], xn[:, 1], s=1, alpha=0.2)
        axes[r, c].set_title(
            f"{name} [{lab}]\non support: {in_support(xx):.3f}", fontsize=10
        )
for ax in axes.flat:
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
plt.tight_layout()
plt.show()


torch.manual_seed(2)
m = 4000
grid = np.linspace(-5, 5, 1001)
fig, axes = plt.subplots(1, 4, figsize=(20, 3.5))
for ax, c in zip(axes, [-3.0, -1.0, 1.0, 3.0]):
    init = torch.stack([torch.full((m,), c, device=device), base(m)[:, 1]], dim=1)
    y_ode = grab(sample_ode(eta, inpainting_path(obsX), init, n_step=n_step)[:, 1])
    y_sde = grab(
        sample_sde(
            eta, inpainting_path(obsX), init, n_step=n_step_sde, eps_bar=eps_bar
        )[:, 1]
    )
    true = 0.25 * (
        (grid >= -4) & (grid < 4) & (((np.floor(c / 2) + np.floor(grid / 2)) % 2) == 0)
    )
    ax.hist(y_ode, bins=80, range=(-5, 5), density=True, alpha=0.5, label="ODE")
    ax.hist(y_sde, bins=80, range=(-5, 5), density=True, alpha=0.5, label="SDE")
    ax.plot(grid, true, "k", lw=1.5, label=r"true $p(Y\mid X)$")
    ax.set_title(f"X observed = {c}")
    ax.legend(fontsize=8)
plt.show()


_, logpx = log_likelihood(eta, base(5000))
print(
    f"log p (probability flow, unconditional path): {logpx.mean().item():.3f}    true value: {-np.log(32):.3f}"
)


class BFromEta(torch.nn.Module):
    def __init__(self, eta):
        super().__init__()
        self.eta = eta

    def forward(self, x, t):
        a = (1 - t) * torch.ones_like(x)
        return -self.eta(x, a)


class EtaSIFromEta(torch.nn.Module):
    def __init__(self, eta):
        super().__init__()
        self.eta = eta

    def forward(self, x, t):
        a = (1 - t) * torch.ones_like(x)
        return -HadamardInterpolant.eta0(self.eta(x, a), a, x)


b_si, eta_si = BFromEta(eta), EtaSIFromEta(eta)
with torch.no_grad():
    xt, t = base(5), torch.tensor([0.3], device=device)
    a = (1 - t) * torch.ones_like(xt)
    e = eta(xt, a)
    print(
        (
            a * HadamardInterpolant.eta0(e, a, xt)
            + (1 - a) * HadamardInterpolant.eta1(e, a, xt)
            - xt
        )
        .abs()
        .max()
    )
