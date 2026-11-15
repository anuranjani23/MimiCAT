# MimiCAT-ReImplementation: Name-Free Pose Transfer between Rigged Characters

A from-scratch re-implementation of the two-stage pipeline described in the MiMiCAT paper
(correspondence transformer **G** and pose-transfer transformer **H**), built without any public
training code, on the PokeAnimDB creature dataset.

At inference the model sees **no joint names and no category labels**. It receives only geometry
(rest-pose joints, skeleton tree, skinning weights, a vertex cloud) for a source and a target
character and decides which source joints drive which target joints, even when the two skeletons
have different topologies and joint counts (3 to 174 joints in the dataset).

> Everything in this README was run on a laptop CPU. All numbers come from the logs in `outputs/`.

---

## 1. Pipeline at a glance

```
 source character (rest pose, skeleton, skin weights, vertex cloud)  ─┐
                                                                      ├─► G ─► soft correspondence M (Js×Jt)
 target character (same inputs)                                      ─┘            │
                                                                                   ▼
 source motion R_src (per-joint rotations) ──► weights_from_M ──► blend rotations ──► R_init (target)
                                                                                   │
                                                                      (optional) H refines ──► R_tgt
                                                                                   │
                                                              LBS ──► posed target mesh, video
```

* **Stage I – G** predicts a joint-to-joint correspondence (with a dustbin for "no counterpart").
* **Stage II – H** refines the target rotations that the correspondence produced.
* Joint **names are used only to build training targets and one evaluation metric**, never as model input.

## 2. Demo

Left to right: source motion, **hard** (geometric Hungarian baseline), **g** (ours), **oracle**
(label-derived correspondence, an upper-bound reference). The number in each label is how many
target joints that variant drives.

Successful Demos:

https://github.com/user-attachments/assets/ca65a7ca-af71-4411-a975-55815ba8a5a0


https://github.com/user-attachments/assets/fd33a4fd-c43c-4484-868d-aa9bed182c43


https://github.com/user-attachments/assets/6b65c622-d11f-4cea-81f1-81e1ba21c057


https://github.com/user-attachments/assets/8d3a66be-ef4b-4c21-a390-4e9ec16e9d02





How the pairs were chosen: `viz/pick_pairs.py` scores 100 random test pairs (seed 7) by G's label
agreement and picks the best ("success"), the median ("typical"), the worst ("failure"), and the
pair where G beats the baseline most. Translation is zeroed, so characters move in place. 

| Pair (src → tgt) | hard | G | oracle | target joints |
|---|---|---|---|---|
| `pm0708_14_00 → pm0449_00_00` (G beats hard most) | 37 | 15 | 30 | 37 |
| `pm0125_00_00 → pm0863_00_00` (success) | 30 | 21 | 28 | 30 |
| `pm0402_01_00 → pm1056_11_00` (success) | 36 | 15 | 19 | 65 |
| `pm0755_00_00 → pm0203_01_00` (typical) | 36 | 29 | 33 | 36 |
| `pm0863_00_00 → pm0772_00_00` (failure) | 30 | 21 | 30 | 37 |

*Driven target joints per variant.* The geometric baseline drives almost every joint, but driving
more joints is not better: it matches by geometry alone and gets many semantics wrong (see §4).

## 3. Results

### 3.1 Stage I: correspondence quality (test split, 200 pairs, 69 test characters)

`python -m mimicat.eval.eval_g --ckpt checkpoints/g.pt --split test`

| Method | acc | acc (matched joints) | acc (dustbin) | F1 |
|---|---|---|---|---|
| Hungarian (geometric baseline) | 0.348 | 0.391 | 0.205 | 0.311 |
| **G (ours)** | **0.667** | **0.657** | **0.663** | **0.571** |

By pair type (`same_plan` = identical body-plan tag sets; only a few such pairs exist):

| | cross_plan acc / F1 | same_plan acc / F1 |
|---|---|---|
| Hungarian | 0.342 / 0.300 | 0.384 / 0.384 |
| G | 0.667 / 0.560 | 0.667 / 0.644 |

Metrics are the per-direction (source→target and target→source) averages from `eval/metrics.py`;
"any acceptable answer counts" when the label target has several valid partners.

Loss-weight ablation (`--match-w`, weight on matched vs dustbin rows), test split, same 200 pairs:

| `--match-w` | steps | test F1@thr | test acc | test F1 | val F1@thr |
|---|---|---|---|---|---|
| 1 | 2000 | 0.542 | 0.659 | 0.554 | 0.508 |
| **2 (default, used)** | 3000 | 0.551 | 0.667 | 0.571 | 0.517 |
| 3 | 2000 | 0.541 | 0.653 | 0.561 | 0.511 |

The three settings differ by about 0.01 F1 with a single seed each, which is within what I would
expect from noise. The default was kept because it had the best validation F1, not because the
difference is established.

### 3.2 Stage II: pose transfer (test split, 100 pairs, 4 frames each)

`python -m mimicat.eval.evaluate --g-ckpt checkpoints/g.pt --h-ckpt checkpoints/h.pt --h-oracle-ckpt checkpoints/h_oracle.pt`

| Variant | cycle error ± SEM (lower = better) | label agreement, deg (lower = better) |
|---|---|---|
| hard | 0.0209 ± 0.0048 | 12.45 |
| hard + H | 0.0229 ± 0.0048 | 12.57 |
| oracle (reference) | 0.0478 ± 0.0052 | 2.45 |
| oracle + H(oracle-trained) | 0.0477 ± 0.0052 | 2.47 |
| **G** | 0.0548 ± 0.0054 | **6.74** |
| G + H | 0.0539 ± 0.0054 | 6.67 |
| G + H (oracle-trained) | 0.0545 ± 0.0054 | 6.71 |
| G + H (label-supervised, `h_agree`) | 0.0541 ± 0.0054 | 6.68 |
| G + Sinkhorn | 0.0768 ± 0.0062 | 10.41 |
| G + Hungarian on G's scores | 0.0564 ± 0.0067 | 6.09 |

Full tables, including same-/cross-plan splits, are in `outputs/results.csv` and
`outputs/results_hagree.csv`.

**What the table does and does not show**

* **G roughly halves the baseline's label-agreement error** (12.45° → 6.74°). This is the main
  Stage II result.
* **Cycle error ranks the methods backwards** (hard "best", oracle worse than hard). It rewards
  invertible mappings, not semantically correct ones, so treat it as a secondary diagnostic.
  See §6.
* **The oracle row is circular**: label agreement is computed against the same name-derived labels
  the oracle uses. It is an upper-bound reference, not a result.
* **H adds nothing measurable.** Differences between G and G + H (any H variant) are far below one
  standard error. Validation curves agree (no-H 0.0473 → H 0.0461 cycle, label agreement 6.5° → 6.5°).
* **G + Sinkhorn is clearly worse** (10.41°), consistent with unit-capacity Sinkhorn conflicting with
  the many-to-one label targets.
* **G + Hungarian has the lowest label agreement of the non-oracle methods** (6.09°) but a slightly
  higher cycle error. It was not tested for significance and is reported as an observation.
* Only 8 of the 100 test pairs are `same_plan`; do not draw conclusions from those rows.
* One seed per configuration, no paired significance tests.

## 4. Setup

```bash
conda create -n mimicat python=3.10 -y && conda activate mimicat
pip install numpy scipy torch tqdm matplotlib pillow trimesh pyrender opencv-python
```

(`pyrender`, `trimesh` and `opencv-python` are only needed for `viz/render_video.py`.)
Run every command **from the repository root with `python -m`** (the package uses relative imports).

### Data

Download PokeAnimDB as described in the upstream MimiCAT instructions and place it so that:

```
data/PokeAnimDB/data/<sample>/<sample>.npz          # v, f, joints, joint_names, parents, weights
data/PokeAnimDB/data/<sample>/motion/<motion>.npz   # body_pose (T, J, 3) rotvec, transl (T, 3)
```

Dataset facts (from `scripts/inspect_dataset.py`): 961 characters, 33,933 motions, 3 to 174 joints
per character (median 45).

```bash
python scripts/inspect_sample.py        # sanity check one sample
python scripts/inspect_dataset.py       # dataset statistics
python scripts/check_labels.py          # how well name parsing covers the joints
python scripts/make_subset.py --k-per-tag 6 --j-min 15 --j-max 80   # writes configs/subset.json
```

`make_subset.py` picks characters per body-plan tag (`limbless`, `arm+leg`, `feeler`, `tail`,
`winged`, `leg_only`, `arm_only`) with 15 to 80 joints and reports 42 anchor characters split
30/6/6 into train/val/test. The training scripts report 324 / 69 / 69 characters per split (see
`train/common.py::names_in` for how the split file is resolved).

## 5. Repository layout

```
mimicat/
  paths.py                      DATA_ROOT
  data/
    dataset.py                  index, motion loading
    tensors.py                  CharStore: per-character tensors (features, points, skin, motion sampling)
  models/
    tokens.py                   FEAT_DIM = 40 name-free joint features, vertex subsampling
    common.py                   SelfBlock, CrossBlock, PointShape (tiny PointNet)
    corr_transformer.py         G
    pose_transformer.py         H
  correspondence/
    labels.py                   joint-name parsing, capacity-aware target M_gt (supervision only)
    soft.py                     softmax / Sinkhorn, losses, threshold calibration
    hard.py                     Hungarian baseline (with dustbin), one-to-one assignment on scores
    provider.py                 CorrProvider (oracle | hard | g | g_sinkhorn | g_hungarian), GTCache
    transfer.py                 6D rotation helpers, rotation blending
  geometry/lbs.py               linear blend skinning
  train/
    common.py                   seeding, split loading, pair sampling, checkpoint I/O
    train_g.py                  Stage I
    train_h.py                  Stage II
  eval/
    eval_g.py                   correspondence metrics (G vs Hungarian)
    evaluate.py                 transfer ablation table
    cycle.py                    pose transfer + cycle error
    metrics.py                  correspondence and label-agreement metrics
  viz/
    pick_pairs.py               choose demo pairs by label agreement
    render_transfer.py          still-frame figure (matplotlib point clouds)
    render_video.py             side-by-side mp4 (pyrender)
scripts/                        dataset inspection, label check, subset generation
configs/subset.json
checkpoints/                    g.pt (+ g.pt.thr.json), h.pt, h_oracle.pt, h_agree.pt, g_mw1.pt, g_mw3.pt
outputs/                        CSV results, logs, figures, videos
docs/                           demo gifs/videos used in this README
```

## 6. Method

### 6.1 Name-free joint features (`models/tokens.py`, 40 dimensions)

Coordinates are normalised by the character's bounding box. Columns 0 to 17 are position, offset
to parent, bone length, depth, child count, leaf flag, subtree size, skin mass, skin-centroid
offset and skin spread. Columns 18 to 39 add skeleton-topology and symmetry descriptors: unit bone
direction, |position|, height to deepest leaf, leaf fraction, path length from root, hops since
last branch point, branch flag, sibling count and rank, parent child count and bone length, bend
relative to the parent bone, mean child bone length and bend, subtree spatial extent, and position
relative to the root joint.

### 6.2 Supervision from names (`correspondence/labels.py`)

Joint names are parsed into side, part, family (e.g. forelimb, tail, feeler), chain index and
auxiliary flags. A pairwise affinity (exact part > same family > cross-limb, with side and chain-position
penalties) feeds a **capacity-aware dustbin Sinkhorn** that yields the target `M_aug`. Capacity
means several source joints may legitimately map to one target joint (e.g. four `waist_*` joints to
one `waist`). Over 200 random pairs about 74% of source and 73% of target joints have a label match.

These labels come from hand-written heuristics, not ground truth. Both training targets and the
label-agreement metric inherit their errors.

### 6.3 Correspondence transformer G (`models/corr_transformer.py`, 1.94M parameters)

* Joint tokens from `[raw features, pair-standardised features]` plus 2 shape tokens from a small PointNet.
* 3 interleaved self-attention (shared weights across the two characters) and cross-attention layers.
* Cosine affinity with a learnable matrix `A` and temperature, plus **per-joint dustbin logits**.
* Loss: row/column cross-entropy against `M_aug`, matched rows up-weighted (`--match-w 2`).
  EMA weights are evaluated and saved.
* Inference matrix `M = sqrt(p_row * p_col)`; a **match threshold (0.30) is calibrated on validation
  F1** and stored in `checkpoints/g.pt.thr.json`. Always load it with `load_threshold(...)`.

### 6.4 Pose transfer and H (`correspondence/transfer.py`, `models/pose_transformer.py`)

`weights_from_M` turns `M` into per-target-joint weights over source joints; source rotations are
blended with those weights and projected back to SO(3) (6D Gram-Schmidt); unmatched target joints
keep the identity. H (0.36M parameters) refines the rotations as a residual in 6D, with a
zero-initialised output so it starts as a no-op. H losses: cycle consistency (A → B → A vertex
error), identity transfer (A → A), an L2 prior on the residual, and an optional label-agreement
term (`--w-agree`, used only in one ablation).

### 6.5 Metrics (`eval/metrics.py`, `eval/cycle.py`)

* **Correspondence acc / F1**: argmax prediction against the label target, both directions averaged.
* **Label agreement (deg)**: mean geodesic angle between a target joint's rotation and its
  label-matched source joint's rotation (confident label matches only). It checks semantics but favours rotation copying.
* **Cycle error**: A → B → A vertex error / bounding-box size. It does **not** measure correct
  transfer. In our test results the one-to-one hard baseline has the lowest cycle error
  (0.0209) and the worst label agreement among non-H variants of its kind (12.45°), while the
  many-to-one oracle pays in cycle error (0.0478) despite being semantically right. A plausible
  reason is that many-to-one mappings are lossy on the return trip; this was not isolated by an
  experiment. Report both metrics and say which one you trust.

## 7. Reproducing the results

Commands exactly as run (laptop CPU; timings from the logs).

```bash
# Stage I: G (about 22 min)
python -m mimicat.train.train_g --steps 3000 --batch 16 --out checkpoints/g.pt
python -m mimicat.eval.eval_g --ckpt checkpoints/g.pt --split test --out outputs/g_results.csv

# loss-weight ablations (about 14 min each)
python -m mimicat.train.train_g --steps 2000 --batch 16 --match-w 1 --out checkpoints/g_mw1.pt
python -m mimicat.train.train_g --steps 2000 --batch 16 --match-w 3 --out checkpoints/g_mw3.pt

# Stage II: H on oracle correspondences (about 13 min), then on G's (about 15 min)
python -m mimicat.train.train_h --corr oracle --steps 2500 --out checkpoints/h_oracle.pt
python -m mimicat.train.train_h --corr g --g-ckpt checkpoints/g.pt --steps 2500 --out checkpoints/h.pt

# label-supervised H ablation (about 9 min)
python -m mimicat.train.train_h --corr g --g-ckpt checkpoints/g.pt --steps 1500 \
  --w-agree 1.0 --w-prior 0.05 --lr 5e-4 --out checkpoints/h_agree.pt

# transfer evaluation tables
python -m mimicat.eval.evaluate --g-ckpt checkpoints/g.pt --h-ckpt checkpoints/h.pt \
  --h-oracle-ckpt checkpoints/h_oracle.pt --out outputs/results.csv
python -m mimicat.eval.evaluate --g-ckpt checkpoints/g.pt --h-ckpt checkpoints/h_agree.pt \
  --h-oracle-ckpt checkpoints/h_oracle.pt --out outputs/results_hagree.csv
```

Figures and videos:

```bash
python -m mimicat.viz.pick_pairs --g-ckpt checkpoints/g.pt --n 80        # prints render commands

python -m mimicat.viz.render_transfer --src pm0125_00_00 --tgt pm0863_00_00 --frames 6 \
  --variants hard,g,g_hungarian,oracle --g-ckpt checkpoints/g.pt --out outputs/demo_success.png

python -m mimicat.viz.render_video --src pm0125_00_00 --tgt pm0863_00_00 \
  --variants hard,g,oracle --g-ckpt checkpoints/g.pt --out outputs/video_success.mp4
python -m mimicat.viz.render_video --src pm0863_00_00 --tgt pm0772_00_00 \
  --variants hard,g,oracle --g-ckpt checkpoints/g.pt --out outputs/video_failure.mp4

# web-friendly copies for this README (needs ffmpeg)
mkdir -p docs
for n in success failure; do
  ffmpeg -y -i outputs/video_$n.mp4 \
    -vf "fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" \
    -loop 0 docs/demo_$n.gif
done
ffmpeg -y -i outputs/video_success.mp4 -c:v libx264 -pix_fmt yuv420p -crf 23 \
  -movflags +faststart docs/demo_success.mp4
```

`render_video.py` writes `mp4v` files, which many browsers and slide programs cannot play. The
ffmpeg lines above produce H.264 / GIF copies that do.

### Key hyperparameters

| | G | H |
|---|---|---|
| width / depth | d=128, 3 self + 3 cross layers, 4 heads, dropout 0.1 | d=64, 3 layers, 4 heads |
| optimiser | AdamW, lr 5e-4, wd 1e-2, 300 warmup steps, cosine | AdamW, lr 3e-4, wd 1e-4, 100 warmup steps, cosine |
| batch | 16 pairs | 4 pairs × 4 frames |
| loss weights | `--match-w 2`, `--ot-w 0` | cycle 1, `--w-id 0.5`, `--w-prior 0.1`, `--w-agree 0` |
| other | EMA 0.995, val 150 pairs every 250 steps, checkpoint on val F1@thr | val 40 pairs every 250 steps, checkpoint on val cycle error |
| seed | 0 | 0 |

## 8. Deviations from the paper

1. No public training code: losses, schedules and hyperparameters are our own, guided by the
   paper's description (two stages, correspondence pretraining, cycle consistency, pose prior).
2. Ground-truth joints and skin weights replace the paper's skin predictor; a small PointNet over a
   512-point vertex subsample replaces the pretrained shape encoder.
3. G is supervised with **name-derived, capacity-aware** targets (row/column cross-entropy, geometric-mean
   inference matrix); Sinkhorn is kept only as an ablation.
4. The pose prior is a plain L2 penalty on H's residual; rotation blending uses Gram-Schmidt
   projection rather than SVD.
5. G additionally uses topology features, per-joint dustbin logits, interleaved attention, a weighted
   loss and a validation-calibrated threshold. These were introduced together and **not ablated individually**.

## 9. Scope and limitations

* **"Name-free" is about inference, not supervision.** Names build the training targets and the
  label-agreement metric. Errors in the name heuristics cap what G can learn and what the metric can show.
* **Dataset scope.** Everything is trained and tested on one creature dataset. Generalisation to other
  rig styles (e.g. humanoid retargeting benchmarks) was not tested.
* There is **no ground-truth target motion**, so transfer quality is measured only indirectly
  (label agreement, cycle error, visual inspection).
* Oracle numbers are circular; cycle error is not a faithful quality measure (§6.5).
* H gives no measurable improvement, even when it is given a direct label-agreement loss.
* Single seed per configuration; no paired significance tests; match threshold tuned on the validation
  split only.
* Videos are qualitative, hand-selected extremes from a random sample, with translation zeroed.
* `pm0772` is the target of both failure examples we looked at. We did not investigate whether it is
  a systematic failure for a body plan.

## 10. Where this can go: toward category-free pose transfer

The design already avoids category labels and joint names at inference, so the natural next steps are:

* replace name heuristics with supervision that does not depend on naming (e.g. motion-statistics or
  deformation-based self-supervision) so G is not capped by the labels;
* evaluate on rigs outside this dataset (humanoids, quadrupeds from other sources) to test whether
  "category-free" holds beyond PokeAnimDB;
* give H a training signal that actually involves target-pose quality rather than only reversibility;
* study the one-to-one assignment on G's scores (`g_hungarian`), which had the best label agreement of
  the non-oracle methods.

## 11. Troubleshooting

* `ImportError: attempted relative import with no known parent package`: run modules from the repo root
  with `python -m mimicat.<package>.<module>`, not `python path/to/file.py`.
* A rendered character lies on its side: add `--up z` (`render_transfer.py` / `render_video.py`).
* A video will not play in a browser or slide deck: use the H.264 copy from the ffmpeg step above.
* Old checkpoints fail to load after changing `FEAT_DIM` or G's architecture: retrain and delete stale
  feature caches.

## 12. Credits

Based on the MiMiCAT paper and its public demo code; PokeAnimDB dataset from the upstream authors.
All training, labelling, evaluation and rendering code in this repository is a re-implementation.
