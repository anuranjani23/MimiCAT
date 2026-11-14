"""Pose transfer transformer H (Stage II): refines the correspondence-initialised target rotations
using target shape/keypoint tokens and cross-attention to the source pose."""
import torch
import torch.nn as nn

from .common import SelfBlock, CrossBlock, PointShape
from .tokens import FEAT_DIM
from ..correspondence.transfer import matrix_to_rot6d, rot6d_to_matrix, IDENT6


class PoseTransformer(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, d=64, heads=4, n_layers=3, n_shape=2):
        super().__init__()
        self.cfg = dict(feat_dim=feat_dim, d=d, heads=heads, n_layers=n_layers, n_shape=n_shape)
        self.t_in = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU(), nn.Linear(d, d))
        self.s_in = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU(), nn.Linear(d, d))
        self.pose_t = nn.Linear(6 + 1, d)
        self.pose_s = nn.Linear(6, d)
        self.shape = PointShape(d, n_shape)
        self.self_blocks = nn.ModuleList([SelfBlock(d, heads) for _ in range(n_layers)])
        self.cross_blocks = nn.ModuleList([CrossBlock(d, heads) for _ in range(n_layers)])
        self.out = nn.Linear(d, 6)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start at the initialisation
        self.register_buffer("ident6", torch.tensor(IDENT6))

    def forward(self, tf, tp, sf, sp, R_src, R_init, matched):
        """tf/sf: (Jt,F)/(Js,F) features; tp/sp: point clouds; R_src (T,Js,3,3);
        R_init (T,Jt,3,3); matched (Jt,) -> R_out (T,Jt,3,3), delta (T,Jt,6)"""
        T, Jt = R_init.shape[0], tf.shape[0]
        r6_src = matrix_to_rot6d(R_src)
        r6_init = matrix_to_rot6d(R_init)
        flag = matched.to(r6_init.dtype)[None, :, None].expand(T, Jt, 1)
        xt = self.t_in(tf)[None] + self.pose_t(torch.cat([r6_init - self.ident6, flag], -1))
        st = self.shape(tp)[None].expand(T, -1, -1)
        xs = self.s_in(sf)[None] + self.pose_s(r6_src - self.ident6)
        x = torch.cat([xt, st], 1)
        for sb, cb in zip(self.self_blocks, self.cross_blocks):
            x = cb(sb(x), xs)
        delta = self.out(x[:, :Jt])
        return rot6d_to_matrix(r6_init + delta), delta
