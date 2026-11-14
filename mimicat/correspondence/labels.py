"""Joint-name parsing and the ground-truth correspondence target M_gt.

Names are used ONLY to build supervision targets. Models never see names.
"""
import re, collections
from dataclasses import dataclass
import numpy as np

SIDES = {"left", "right", "center"}
AUX = {"sub", "off", "attach", "roll", "offset", "twist", "root"}
ALIASES = {"ceeler": "feeler", "jow": "jaw"}

FAMILIES = {
    "root":     {"origin", "rigroot"},
    "pelvis":   {"waist", "hips", "crotch", "pelvis"},
    "torso":    {"spine", "chest", "body", "belly"},
    "neck":     {"neck"},
    "head":     {"head", "skull"},
    "jaw":      {"jaw", "teeth", "tongue", "mouth", "beak"},
    "eye":      {"eyelid", "eyelash", "eyebrow", "eyeball", "eye"},
    "ear":      {"ear"},
    "nose":     {"nose"},
    "horn":     {"horn"},
    "forelimb": {"shoulder", "arm", "hand", "wing", "index", "middle", "ring",
                 "thumb", "pinky", "finger", "fingerbase"},
    "hindlimb": {"leg", "foot", "toe", "thigh", "knee"},
    "claw":     {"claw"},
    "tail":     {"tail"},
    "feeler":   {"feeler", "antenna", "tentacle"},
    "hair":     {"hair"},
    "cloth":    {"skirt", "cloth"},
}
FAMILY_OF = {t: f for f, ts in FAMILIES.items() for t in ts}
LIMBS = {"forelimb", "hindlimb"}
EXACT_ONLY = {"root", "junk", "other"}


@dataclass(frozen=True)
class Label:
    side: str
    part: str
    family: str
    index: int | None
    variant: str
    aux: bool


def parse_name(name):
    n = str(name).lower()
    if re.fullmatch(r"pm\d+(_\d+)*", n):
        return Label("none", "rigroot", "root", None, "", False)
    if n.startswith("point") or n in ("raycast", "ground"):
        return Label("none", n, "junk", None, "", False)
    toks = [ALIASES.get(t, t) for t in n.split("_")]
    side = next((t for t in toks if t in SIDES), "none")
    index = next((int(t) for t in reversed(toks) if t.isdigit()), None)
    aux = any(t in AUX for t in toks)
    variant = next((t for t in toks if len(t) == 1 and t.isalpha()), "")
    base = [t for t in toks
            if t not in SIDES and t not in AUX
            and not t.isdigit() and not (len(t) == 1 and t.isalpha())]
    part = "_".join(base) or "unknown"
    family = next((FAMILY_OF[t] for t in base if t in FAMILY_OF), "other")
    return Label(side, part, family, index, variant, aux)


def chain_positions(labels):
    groups = collections.defaultdict(list)
    for i, l in enumerate(labels):
        if l.index is not None and not l.aux:
            groups[(l.side, l.part, l.variant)].append(i)
    pos = np.full(len(labels), np.nan, np.float32)
    for idxs in groups.values():
        if len(idxs) > 1:
            order = sorted(idxs, key=lambda i: labels[i].index)
            for r, i in enumerate(order):
                pos[i] = r / (len(order) - 1)
    return pos


@dataclass
class LabelCfg:
    w_side: float = 0.4
    w_family: float = 0.3
    w_limb: float = 0.1
    w_aux: float = 0.5
    sigma: float = 0.35
    w_dust: float = 0.10     # dustbin score
    tau: float = 0.05        # kernel temperature: K = exp(score / tau)


def _pair_affinity(a, b, pa, pb, c):
    if a.family in EXACT_ONLY or b.family in EXACT_ONLY:
        if a.part != b.part:
            return 0.0
        s = 1.0
    elif a.part == b.part:
        s = 1.0
    elif a.family == b.family:
        s = c.w_family
    elif a.family in LIMBS and b.family in LIMBS:
        s = c.w_limb
    else:
        return 0.0
    if "none" not in (a.side, b.side) and "center" not in (a.side, b.side) and a.side != b.side:
        s *= c.w_side
    if a.aux or b.aux:
        s *= c.w_aux
    if not (np.isnan(pa) or np.isnan(pb)):
        s *= float(np.exp(-abs(pa - pb) / c.sigma))
    return s


def sinkhorn_np(K, a, b, n_iters=500):
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(n_iters):
        u = a / (K @ v)
        v = b / (K.T @ u)
    return u[:, None] * K * v[None, :]


def build_M_gt(names_src, names_tgt, cfg=LabelCfg()):
    """Capacity-aware dustbin Sinkhorn.
    r_i / c_j = how many partners joint i / j can legitimately have (>=1), so that e.g. 4 source
    'waist' joints can all map to 1 target 'waist' without being forced into the dustbin.
    Returns M (Js,Jt), M_aug (Js+1,Jt+1), r (Js,), c (Jt,).
    Row i is matched iff M[i].sum()/r[i] >= 0.5."""
    ls = [parse_name(n) for n in names_src]
    lt = [parse_name(n) for n in names_tgt]
    ps, pt = chain_positions(ls), chain_positions(lt)
    Js, Jt = len(ls), len(lt)
    A = np.array([[_pair_affinity(ls[i], lt[j], ps[i], pt[j], cfg)
                   for j in range(Jt)] for i in range(Js)], dtype=np.float64)
    strong = A >= cfg.w_family
    r = np.maximum(1, strong.sum(1)).astype(np.float64)
    c = np.maximum(1, strong.sum(0)).astype(np.float64)
    K = np.full((Js + 1, Jt + 1), np.exp(cfg.w_dust / cfg.tau))
    K[:Js, :Jt] = np.exp(A / cfg.tau)
    a = np.r_[r, c.sum()]
    b = np.r_[c, r.sum()]
    M_aug = sinkhorn_np(K, a, b)
    return (M_aug[:Js, :Jt].astype(np.float32), M_aug.astype(np.float32),
            r.astype(np.float32), c.astype(np.float32))


def transfer_weights(M, min_mass=0.2):
    col = M.sum(0)
    W = (M / np.maximum(col, 1e-8)).T
    return W.astype(np.float32), col >= min_mass
