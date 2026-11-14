import numpy as np, collections
from pathlib import Path

root = Path("data/PokeAnimDB/data")
key_sets, jcounts, nmot = collections.Counter(), [], []
samples = sorted(p for p in root.iterdir() if p.is_dir())
for s in samples:
    z = np.load(s / f"{s.name}.npz", allow_pickle=True)
    key_sets[tuple(sorted(z.files))] += 1
    jcounts.append(len(z["parents"]))
    nmot.append(len(list((s / "motion").glob("*.npz"))))
print("samples:", len(samples), "| total motions:", sum(nmot))
print("joints per char: min", min(jcounts), "max", max(jcounts),
      "median", int(np.median(jcounts)))
for ks, c in key_sets.items():
    print(c, "chars have keys:", ks)