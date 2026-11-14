"""Print the file tree and array shapes of one character.   python scripts/inspect_sample.py [name]"""
import sys, numpy as np
from pathlib import Path
from mimicat.paths import DATA_ROOT

name = sys.argv[1] if len(sys.argv) > 1 else "pm0025_17_00"
d = Path(DATA_ROOT) / name
print("root:", DATA_ROOT)
for p in sorted(d.rglob("*")):
    if p.is_file() and p.parent.name != "motion":
        print(" ", p.relative_to(d))
motions = sorted((d / "motion").glob("*.npz"))
print("motion files:", len(motions))

def dump(path):
    z = np.load(path, allow_pickle=True)
    print(f"\n== {path.name} ==")
    for k in z.files:
        v = z[k]
        print(f"{k:20s} {str(v.dtype):10s} {v.shape}")

dump(d / f"{name}.npz")
dump(motions[0])
