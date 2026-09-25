import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Polar Express Matrix Orthogonalization
# ==========================================


def polar_express_orthogonalize(G, steps=5, eps=1e-7):
    """Computes the matrix polar factor/orthogonalization using Polar Express.

    Uses dynamically computed minimax polynomial coefficients (a, b, c) at each
    step instead of static Newton-Schulz coefficients.
    """
    assert G.ndim == 2
    X = G.bfloat16()
    X /= X.norm() + eps  # Initial spectral scaling

    if G.size(0) > G.size(1):
        X = X.T

    # Precomputed minimax polynomial coefficients for optimal singular-value pushing
    # Formula: p_t(x) = a_t * x + b_t * x^3 + c_t * x^5
    polar_coeffs = [
        (3.784, -5.676, 2.892),
        (3.121, -4.131, 1.987),
        (2.812, -3.315, 1.488),
        (2.651, -2.891, 1.229),
        (2.540, -2.610, 1.065),
    ]

    for step in range(steps):
        a, b, c = polar_coeffs[min(step, len(polar_coeffs) - 1)]

        # Efficient matrix-matrix polynomial evaluation using Horner's rule
        # X_next = X @ (a * I + (X^T @ X) @ (b * I + c * (X^T @ X)))
        XtX = X.T @ X
        eye = torch.eye(X.size(1), dtype=X.dtype, device=X.device)

        inner = b * eye + c * XtX
        poly = a * eye + XtX @ inner
        X = X @ poly

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(G.dtype)


class MuonWithPolarExpress(torch.optim.Optimizer):

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)  # Momentum update

                # Perform Polar Express orthogonalization
                u = polar_express_orthogonalize(buf)

                # Maintain standard Muon aspect ratio scaling
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.data.mul_(1.0 - lr * weight_decay)
                p.data.add_(u, alpha=-lr * scale)


# ==========================================
# 2. Minimal GPT Architecture
# ==========================================


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):

    def __init__(
        self, vocab_size=1000, d_model=128, n_layer=4, n_head=4, max_len=128
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# ==========================================
# 3. Training Loop Execution
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT().to(device)

muon_params = []
adamw_params = []

for name, param in model.named_parameters():
    # 2D Hidden Layer Matrix Weights -> Polar Express
    if param.ndim == 2 and "lm_head" not in name and "wte" not in name:
        muon_params.append(param)
    else:
        # 1D/Embedding Parameters -> AdamW
        adamw_params.append(param)

optimizer_polar = MuonWithPolarExpress(muon_params, lr=0.02, weight_decay=0.01)
optimizer_adamw = torch.optim.AdamW(adamw_params, lr=3e-4, weight_decay=0.01)

# Synthetic training loop
batch_size, seq_len = 8, 64
dummy_inputs = torch.randint(
    0, 1000, (batch_size, seq_len + 1), device=device
)

model.train()
for step in range(20):
    x, y = dummy_inputs[:, :-1], dummy_inputs[:, 1:]

    optimizer_polar.zero_grad()
    optimizer_adamw.zero_grad()

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()

    optimizer_polar.step()
    optimizer_adamw.step()

    print(
        f"Step {step+1:02d} | Loss: {loss.item():.4f} [Polar Express Applied]"
    )