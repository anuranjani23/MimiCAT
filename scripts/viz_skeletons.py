"""Save skeleton+mesh pictures for the chosen subset.   python scripts/viz_skeletons.py"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mimicat.paths import DATA_ROOT
from mimicat.data.dataset import load_character
from mimicat.viz.skeleton_viz import draw_skeleton

out = Path("outputs/skeletons"); out.mkdir(parents=True, exist_ok=True)
names = list(json.load(open("configs/subset.json")))
for n in names:
    c = load_character(DATA_ROOT, n)
    ax = draw_skeleton(c.joints, c.parents, verts=c.v, names=c.joint_names)
    ax.set_title(f"{n}  J={len(c.joints)}", fontsize=8)
    ax.figure.savefig(out / f"{n}.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
print("wrote", len(names), "images to", out)
