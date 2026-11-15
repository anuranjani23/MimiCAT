"""Pick demo pairs by G's label agreement (deg, lower is better) and print render commands.
    python -m mimicat.viz.pick_pairs --g-ckpt checkpoints/g.pt --n 80
"""
import argparse
import numpy as np
import torch
from tqdm import tqdm

from ..paths import DATA_ROOT
from ..data.tensors import CharStore
from ..correspondence.provider import GTCache, CorrProvider, load_threshold
from ..eval.cycle import transfer_pose
from ..eval.metrics import label_agreement_deg
from ..train.common import load_split, names_in, all_pairs, load_G, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="configs/subset.json")
    ap.add_argument("--g-ckpt", default="checkpoints/g.pt")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = get_device(a.device)
    names = names_in(load_split(a.subset), "test")
    pairs = all_pairs(names)
    rng = np.random.default_rng(7)
    pairs = [pairs[i] for i in rng.choice(len(pairs), min(a.n, len(pairs)), replace=False)]
    store, gtc = CharStore(DATA_ROOT, dev), GTCache(dev)
    G = load_G(a.g_ckpt, dev)
    pg = CorrProvider("g", G=G, thr=load_threshold(a.g_ckpt), gt_cache=gtc, device=dev)
    ph = CorrProvider("hard", gt_cache=gtc, device=dev)

    rows = []
    with torch.no_grad():
        for an, bn in tqdm(pairs, desc="scoring pairs", unit="pair"):
            A, B = store.get(an), store.get(bn)
            R_a = store.sample_motion(A, 4, rng)
            pr_t, pc_t = gtc.get(A, B)
            Rg = transfer_pose(A, B, R_a, pg)[0]
            Rh = transfer_pose(A, B, R_a, ph)[0]
            g, h = label_agreement_deg(R_a, Rg, pr_t, pc_t), label_agreement_deg(R_a, Rh, pr_t, pc_t)
            if g == g and h == h:
                rows.append((g, h, an, bn))
    rows.sort()
    picks = {
        "success (G best)": rows[0],
        "typical (G median)": rows[len(rows) // 2],
        "failure (G worst)": rows[-1],
        "G beats hard most": max(rows, key=lambda r: r[1] - r[0]),
    }
    for k, (g, h, an, bn) in picks.items():
        print(f"\n# {k}: G {g:.1f} deg, hard {h:.1f} deg")
        print(f"python -m mimicat.viz.render_transfer --src {an} --tgt {bn} --frames 6 "
              f"--variants hard,g,g_hungarian,oracle --g-ckpt {a.g_ckpt} "
              f"--out outputs/demo_{k.split()[0]}_{an}_{bn}.png")


if __name__ == "__main__":
    main()