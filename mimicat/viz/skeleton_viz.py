import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_skeleton(joints, parents, ax=None, verts=None, names=None, up="y"):
    """3D skeleton (+ optional mesh points and joint names). up: which data axis points up."""
    ax = ax or plt.figure(figsize=(5, 5)).add_subplot(projection="3d")
    order = {"y": (0, 2, 1), "z": (0, 1, 2)}[up]
    P = lambda a: a[:, order]
    if verts is not None:
        ax.scatter(*P(verts).T, s=0.3, alpha=0.15)
    jp = P(joints)
    for j, p in enumerate(parents):
        if p >= 0:
            ax.plot(*zip(jp[j], jp[p]), c="r", lw=1)
    ax.scatter(*jp.T, c="k", s=8)
    if names is not None:
        for j, n in enumerate(names):
            ax.text(*jp[j], n, fontsize=4)
    ax.set_box_aspect((1, 1, 1))
    return ax
