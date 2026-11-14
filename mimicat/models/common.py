import torch
import torch.nn as nn


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class SelfBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = mlp(d, 4 * d, d)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class CrossBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln_q, self.ln_c, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = mlp(d, 4 * d, d)

    def forward(self, x, ctx):
        q, c = self.ln_q(x), self.ln_c(ctx)
        x = x + self.attn(q, c, c, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class PointShape(nn.Module):
    """Tiny PointNet: vertex cloud (P,3) -> n_shape tokens (n_shape, d). Stand-in for the
    paper's pretrained shape encoder."""
    def __init__(self, d, n_shape=2):
        super().__init__()
        self.n_shape, self.d = n_shape, d
        self.point = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU())
        self.head = nn.Linear(256, n_shape * d)

    def forward(self, pts):
        h = self.point(pts)
        g = torch.cat([h.max(0).values, h.mean(0)], -1)
        return self.head(g).view(self.n_shape, self.d)
