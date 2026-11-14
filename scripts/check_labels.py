"""Sanity-check the name parser and M_gt on random pairs.   python scripts/check_labels.py"""
import numpy as np, random, collections
from pathlib import Path
from mimicat.paths import DATA_ROOT
from mimicat.correspondence.labels import parse_name, build_M_gt

root = Path(DATA_ROOT)
samples = sorted(p for p in root.iterdir() if p.is_dir())
names = {s.name: [str(x) for x in np.load(s / f"{s.name}.npz", allow_pickle=True)["joint_names"]]
         for s in samples}

fam, other = collections.Counter(), collections.Counter()
for ns in names.values():
    for n in ns:
        l = parse_name(n); fam[l.family] += 1
        if l.family == "other": other[l.part] += 1
print("family counts:", dict(fam))
print("top 30 unmapped parts:", other.most_common(30))

random.seed(0)
usable = [k for k, v in names.items() if len(v) >= 8]
stats = []
for _ in range(200):
    a, b = random.sample(usable, 2)
    M, _, r, c = build_M_gt(names[a], names[b])
    stats.append(((M.sum(1) / r >= 0.5).mean(), (M.sum(0) / c >= 0.5).mean()))
s = np.array(stats)
print(f"\nover 200 random pairs: matched src {s[:,0].mean():.2f}, matched tgt {s[:,1].mean():.2f}")

random.seed(1)
for _ in range(3):
    a, b = random.sample(usable, 2)
    M, _, r, c = build_M_gt(names[a], names[b])
    rm, best = M.sum(1) / r, M.argmax(1)
    print(f"\n{a} ({len(names[a])}) -> {b} ({len(names[b])})")
    print("  matched:")
    for i in range(min(30, len(rm))):
        if rm[i] >= 0.5:
            print(f"    {names[a][i]:26s} -> {names[b][best[i]]:26s} frac {rm[i]:.2f}")
    print("  unmatched src:", [names[a][i] for i in range(len(rm)) if rm[i] < 0.5][:20])
    print("  target names :", names[b][:40])
