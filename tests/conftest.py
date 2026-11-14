import numpy as np, pytest
from pathlib import Path
from mimicat.paths import DATA_ROOT
from mimicat.data.dataset import load_character

SAMPLE, MOTION = "pm0025_17_00", "attack01_0"


@pytest.fixture(scope="session")
def sample():
    d = Path(DATA_ROOT) / SAMPLE
    if not (d / f"{SAMPLE}.npz").exists():
        pytest.skip(f"{SAMPLE} not found under {DATA_ROOT}")
    c = load_character(DATA_ROOT, SAMPLE)
    m = np.load(d / "motion" / f"{MOTION}.npz")
    return dict(v=c.v, f=c.f, joints=c.joints, parents=c.parents, weights=c.weights,
                body_pose=m["body_pose"], transl=m["transl"])
