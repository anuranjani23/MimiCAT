import json, random
from pathlib import Path
import numpy as np
import torch

from ..models.corr_transformer import CorrTransformer
from ..models.pose_transformer import PoseTransformer


def get_device(name="cpu"):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return name


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_split(path="configs/subset.json"):
    return json.loads(Path(path).read_text())


def names_in(split, which):
    return sorted(n for n, s in split.items() if s == which)


def sample_pair(rng, names, p_self=0.0):
    a = names[int(rng.integers(len(names)))]
    if rng.random() < p_self:
        return a, a
    b = a
    while b == a:
        b = names[int(rng.integers(len(names)))]
    return a, b


def all_pairs(names):
    return [(a, b) for a in names for b in names if a != b]


def save_ckpt(path, model, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": model.cfg, **extra}, path)


def load_G(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    G = CorrTransformer(**ck["cfg"]).to(device)
    G.load_state_dict(ck["model"]); G.eval()
    return G


def load_H(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    H = PoseTransformer(**ck["cfg"]).to(device)
    H.load_state_dict(ck["model"]); H.eval()
    return H
