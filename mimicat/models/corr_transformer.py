"""Correspondence transformer G (Stage I).

Inputs are geometric joint features + a vertex cloud per character (no names).
Output: augmented logit matrix S_aug (Js+1, Jt+1). Last row/col = dustbin.

Changes vs. v1:
  * pair-wise z-scored features are concatenated with raw features
  * self/cross attention are interleaved (SuperGlue style) instead of stacked
  * MLP projection head
  * per-joint dustbin logits (a joint's "has no counterpart" depends on the joint)
  * larger defaults (d=128, 3 layers)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SelfBlock, CrossBlock, PointShape
from .tokens import FEAT_DIM


class CorrTransformer(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, d=128, heads=4, n_self=3, n_cross=3,
                 n_shape=2, drop=0.1):
        super().__init__()
        self.d, self.n_shape = d, n_shape
        self.cfg = dict(feat_dim=feat_dim, d=d, heads=heads, n_self=n_self,
                        n_cross=n_cross, n_shape=n_shape, drop=drop)

        # raw + pair-standardised features -> 2 * feat_dim
        self.joint_in = nn.Sequential(
            nn.Linear(2 * feat_dim, d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d, d), nn.LayerNorm(d),
        )
        self.shape = PointShape(d, n_shape)
        self.self_blocks = nn.ModuleList([SelfBlock(d, heads) for _ in range(n_self)])
        self.cross_s = nn.ModuleList([CrossBlock(d, heads) for _ in range(n_cross)])
        self.cross_t = nn.ModuleList([CrossBlock(d, heads) for _ in range(n_cross)])

        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.A = nn.Parameter(torch.eye(d))                       # learnable affinity
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        # per-joint dustbin logit (shared head for source & target) + corner
        self.dust_head = nn.Linear(d, 1)
        nn.init.zeros_(self.dust_head.weight)
        nn.init.constant_(self.dust_head.bias, 2.0)
        self.dust = nn.Parameter(torch.tensor(2.0))

    @staticmethod
    def _pair_standardise(sf, tf):
        allf = torch.cat([sf, tf], 0)
        mu = allf.mean(0, keepdim=True)
        sd = allf.std(0, keepdim=True).clamp_min(1e-3)
        return (sf - mu) / sd, (tf - mu) / sd

    def _tokens(self, feats, zfeats, pts):
        j = self.joint_in(torch.cat([feats, zfeats], -1))
        s = self.shape(pts)
        return torch.cat([j, s], 0)[None]                         # (1, J+n_shape, d)

    def forward(self, sf, sp, tf, tp):
        Js, Jt = sf.shape[0], tf.shape[0]
        zs, zt = self._pair_standardise(sf, tf)
        xs, xt = self._tokens(sf, zs, sp), self._tokens(tf, zt, tp)

        L = max(len(self.self_blocks), len(self.cross_s))
        for i in range(L):
            if i < len(self.self_blocks):                         # shared weights
                blk = self.self_blocks[i]
                xs, xt = blk(xs), blk(xt)
            if i < len(self.cross_s):
                xs, xt = self.cross_s[i](xs, xt), self.cross_t[i](xt, xs)

        hs = self.proj(xs[0, :Js])
        ht = self.proj(xt[0, :Jt])
        gs, gt = F.normalize(hs, dim=-1), F.normalize(ht, dim=-1)

        S = self.logit_scale.exp().clamp(max=100.0) * (gs @ self.A @ gt.T)   # (Js, Jt)
        ds = self.dust_head(hs).squeeze(-1)                                   # (Js,)
        dt = self.dust_head(ht).squeeze(-1)                                   # (Jt,)

        top = torch.cat([S, ds[:, None]], 1)                                  # (Js, Jt+1)
        bot = torch.cat([dt[None, :], self.dust.view(1, 1)], 1)               # (1,  Jt+1)
        return torch.cat([top, bot], 0)                                       # (Js+1, Jt+1)