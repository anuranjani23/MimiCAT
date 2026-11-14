"""Pick a morphology-diverse subset + character-level train/val/test split -> configs/subset.json.
    python scripts/make_subset.py [--k-per-tag 4] [--j-min 15] [--j-max 80]"""
import json, argparse, collections
from mimicat.paths import DATA_ROOT
from mimicat.data.dataset import build_index, select_subset, assign_splits

ap = argparse.ArgumentParser()
ap.add_argument("--k-per-tag", type=int, default=4)
ap.add_argument("--j-min", type=int, default=15)
ap.add_argument("--j-max", type=int, default=80)
ap.add_argument("--min-motions", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="configs/subset.json")
a = ap.parse_args()

idx = build_index(DATA_ROOT)
print("characters:", len(idx))
print("tag counts:", dict(collections.Counter(t for m in idx.values() for t in m["tags"])))
sub = select_subset(idx, a.k_per_tag, (a.j_min, a.j_max), a.min_motions, a.seed)
splits = assign_splits(list(sub), a.seed)
for n in sorted(sub, key=lambda n: splits[n]):
    print(f"{n:16s} J={idx[n]['J']:3d} {splits[n]:5s} {sub[n]}")
json.dump(splits, open(a.out, "w"), indent=1)
print(len(sub), "characters ->", dict(collections.Counter(splits.values())), "| wrote", a.out)
