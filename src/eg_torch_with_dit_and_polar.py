import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Polar Express Matrix Orthogonalization
# ==========================================


def polar_express_orthogonalize(G, steps=5, eps=1e-7):
    """Computes the matrix polar factor via Polar Express polynomial iterations."""
    assert G.ndim == 2
    X = G.bfloat16()
    X /= X.norm() + eps

    if G.size(0) > G.size(1):
        X = X.T

    # Minimax polynomial coefficients per step
    polar_coeffs = [
        (3.784, -5.676, 2.892),
        (3.121, -4.131, 1.987),
        (2.812, -3.315, 1.488),
        (2.651, -2.891, 1.229),
        (2.540, -2.610, 1.065),
    ]

    for step in range(steps):
        a, b, c = polar_coeffs[min(step, len(polar_coeffs) - 1)]
        XtX = X.T @ X
        eye = torch.eye(X.size(1), dtype=X.dtype, device=X.device)
        inner = b * eye + c * XtX
        poly = a * eye + XtX @ inner
        X = X @ poly

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(G.dtype)


class PolarExpressOptimizer(torch.optim.Optimizer):

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
                buf.mul_(momentum).add_(g)

                u = polar_express_orthogonalize(buf)

                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.data.mul_(1.0 - lr * weight_decay)
                p.data.add_(u, alpha=-lr * scale)


# ==========================================
# 2. Diffusion Transformer (DiT) Modules
# ==========================================


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal embeddings."""

    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):

    def __init__(self, hidden_dim, n_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_head, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_dim, hidden_dim, bias=False),
        )
        # adaLN-Zero modulation projections (scale, shift, gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=False)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Self-Attention Branch
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP Branch
        norm_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class MiniDiT(nn.Module):

    def __init__(
        self,
        input_size=16,
        patch_size=2,
        in_channels=4,
        hidden_dim=128,
        depth=4,
        num_heads=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2

        # Patchify image/latent map
        self.x_embedder = nn.Linear(
            patch_size * patch_size * in_channels, hidden_dim
        )
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_dim)
        )

        # DiT Blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads) for _ in range(depth)]
        )

        # Output Head
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        )
        self.final_linear = nn.Linear(
            hidden_dim, patch_size * patch_size * in_channels
        )

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(
            shape=(B, C, H // p, p, W // p, p)
        )  # (B, C, h, p, w, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(shape=(B, (H // p) * (W // p), p * p * C))
        return x

    def unpatchify(self, x, H, W):
        B = x.shape[0]
        p = self.patch_size
        c = self.in_channels
        x = x.reshape(shape=(B, H // p, W // p, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(B, c, H, W))

    def forward(self, x, t):
        # 1. Embed input patches & timesteps
        H, W = x.shape[2], x.shape[3]
        x = self.x_embedder(self.patchify(x)) + self.pos_embed
        c = self.t_embedder(t)

        # 2. Run transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # 3. Final norm & output linear layer
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.final_linear(x)

        return self.unpatchify(x, H, W)


# ==========================================
# 3. Training Loop Setup
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniDiT().to(device)

# Partition parameters for Polar Express vs AdamW
polar_params = []
adamw_params = []

for name, param in model.named_parameters():
    # Target 2D projection and MLP matrices inside DiT blocks
    if param.ndim == 2 and not any(
        k in name for k in ["pos_embed", "t_embedder", "final_linear"]
    ):
        polar_params.append(param)
    else:
        # 1D params, embeddings, and output heads use AdamW
        adamw_params.append(param)

optimizer_polar = PolarExpressOptimizer(
    polar_params, lr=0.01, weight_decay=0.01
)
optimizer_adamw = torch.optim.AdamW(adamw_params, lr=1e-4, weight_decay=0.01)

print(
    f"Polar Express params: {len(polar_params)} | AdamW params: {len(adamw_params)}"
)

# Training loop simulation on latent vectors (e.g., 4x16x16)
batch_size = 4
model.train()

for step in range(20):
    optimizer_polar.zero_grad()
    optimizer_adamw.zero_grad()

    # Synthetic target latent images x0
    x0 = torch.randn(batch_size, 4, 16, 16, device=device)

    # Sample random timesteps and Gaussian noise
    t = torch.randint(0, 1000, (batch_size,), device=device)
    noise = torch.randn_like(x0)

    # Linear noise schedule step
    alpha_bar = torch.cos(((t / 1000 + 0.008) / 1.008) * math.pi / 2) ** 2
    alpha_bar = alpha_bar.view(batch_size, 1, 1, 1).to(device)
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

    # Predict noise
    pred_noise = model(xt, t)

    # Simple MSE Loss
    loss = F.mse_loss(pred_noise, noise)
    loss.backward()

    optimizer_polar.step()
    optimizer_adamw.step()

    print(f"Step {step+1:02d} | DiT Noise Loss: {loss.item():.4f}")