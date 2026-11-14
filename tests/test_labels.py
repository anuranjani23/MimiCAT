import numpy as np
from mimicat.correspondence.labels import parse_name, build_M_gt, transfer_weights


def test_parse():
    assert parse_name("left_leg_01").family == "hindlimb"
    l = parse_name("right_upper_eyelid_01")
    assert (l.side, l.part, l.family, l.index) == ("right", "upper_eyelid", "eye", 1)
    assert parse_name("pm0025_11_00").part == "rigroot"
    assert parse_name("spine_02_attach_off").aux
    assert parse_name("left_foot_index_01").family == "hindlimb"
    assert parse_name("ceeler_a").family == "feeler"
    assert parse_name("lower_jow").family == "jaw"
    assert parse_name("front_skirt").family == "cloth"
    assert parse_name("raycast").family == "junk"


NAMES = ["pm1_00", "origin", "waist", "hips", "spine_01", "left_arm_01", "right_arm_01",
         "left_leg_01", "right_leg_01", "tail_01", "tail_02", "head"]


def test_self_match():
    M, _, r, c = build_M_gt(NAMES, NAMES)
    assert (M.argmax(1) == np.arange(len(NAMES))).mean() > 0.8


def test_root_does_not_cross_match():
    M, _, _, _ = build_M_gt(NAMES, NAMES)
    assert M[0, 1] < 0.01 and M[1, 0] < 0.01


def test_unmatched_goes_to_dustbin():
    tgt = ["pm2_00", "origin", "waist", "hips", "head", "left_feeler_a", "right_feeler_a"]
    M, _, r, c = build_M_gt(NAMES, tgt)
    t = NAMES.index("tail_01")
    assert M[t].sum() / r[t] < 0.3
    h = NAMES.index("head")
    assert M[h].sum() / r[h] > 0.5
    W, matched = transfer_weights(M)
    assert W.shape == (len(tgt), len(NAMES))


def test_many_to_one_is_not_dustbinned():
    src = ["pm1_00", "origin", "waist", "waist_a", "waist_b", "waist_c", "hips"]
    tgt = ["pm2_00", "origin", "waist", "hips"]
    M, _, r, c = build_M_gt(src, tgt)
    for k in (2, 3, 4, 5):
        assert M[k].sum() / r[k] > 0.5 and M[k].argmax() == 2
