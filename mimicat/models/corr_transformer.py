"""Correspondence transformer G (Stage I).

Inputs are geometric joint features + a vertex cloud per character (no names).
Output: augmented logit matrix S_aug (Js+1, Jt+1); the last row/col is the dustbin.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SelfBlock, CrossBlock, PointShape
from .tokens import FEAT_DIM


class CorrTransformer(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, d=64, heads=4, n_self=2, n_cross=2, n_shape=2):
        super().__init__()
        self.d, self.n_shape = d, n_shape
        self.cfg = dict(feat_dim=feat_dim, d=d, heads=heads, n_self=n_self,
                        n_cross=n_cross, n_shape=n_shape)
        self.joint_in = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU(), nn.Linear(d, d))
        self.shape = PointShape(d, n_shape)
        self.self_blocks = nn.ModuleList([SelfBlock(d, heads) for _ in range(n_self)])
        self.cross_s = nn.ModuleList([CrossBlock(d, heads) for _ in range(n_cross)])
        self.cross_t = nn.ModuleList([CrossBlock(d, heads) for _ in range(n_cross)])
        self.proj = nn.Linear(d, d)
        self.A = nn.Parameter(torch.eye(d))              # learnable affinity weight A
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.dust = nn.Parameter(torch.tensor(2.0))      # dustbin logit

    def _tokens(self, feats, pts):
        j = self.joint_in(feats)
        s = self.shape(pts)
        return torch.cat([j, s], 0)[None]                # (1, J+n_shape, d)

    def forward(self, sf, sp, tf, tp):
        Js, Jt = sf.shape[0], tf.shape[0]
        xs, xt = self._tokens(sf, sp), self._tokens(tf, tp)
        for blk in self.self_blocks:                     # shared weights for both characters
            xs, xt = blk(xs), blk(xt)
        for cs, ct in zip(self.cross_s, self.cross_t):   # shape-keypoint interaction
            xs, xt = cs(xs, xt), ct(xt, xs)
        gs = F.normalize(self.proj(xs[0, :Js]), dim=-1)
        gt = F.normalize(self.proj(xt[0, :Jt]), dim=-1)
        S = self.logit_scale.exp().clamp(max=100.0) * (gs @ self.A @ gt.T)
        S_aug = self.dust.expand(Js + 1, Jt + 1).clone()
        S_aug[:Js, :Jt] = S
        return S_aug
