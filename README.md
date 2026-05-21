# [CVPR 2026] 😸 MimiCAT: Mimic with Correspondence-Aware Cascade-Transformer for Category-Free 3D Pose Transfer

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green?style=for-the-badge&logo=googlechrome&logoColor=white)](https://mimicat3d.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.18370-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.18370)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-PokeAnimDB-yellow?style=for-the-badge)](https://huggingface.co/datasets/mimicat3d/PokeAnimDB)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue?style=for-the-badge)](https://creativecommons.org/licenses/by-nc/4.0/)

**[Zenghao Chai](https://zenghaochai.com/)<sup>1</sup> · [Chen Tang](https://www.chentang.cc/#)<sup>2</sup> · [Yongkang Wong](https://sites.google.com/site/yongkangwong)<sup>1</sup> · [Xulei Yang](https://dawdleryang.github.io)<sup>3</sup> · [Mohan Kankanhalli](https://www.comp.nus.edu.sg/~mohan/)<sup>1</sup>**

<sup>1</sup>School of Computing, National University of Singapore &nbsp; | &nbsp;
<sup>2</sup>MMLab, The Chinese University of Hong Kong &nbsp; | &nbsp;
<sup>3</sup>Institute for Infocomm Research, A*STAR

</div>


## 📄 Abstract

3D pose transfer aims to transfer the pose-style of a source mesh to a target character while preserving both the target's geometry and the source's pose characteristic. Existing methods are largely restricted to characters with similar structures and fail to generalize to category-free settings (e.g., transferring a humanoid's pose to a quadruped). The key challenge lies in the structural and transformation diversity inherent in distinct character types, which often leads to mismatched regions and poor transfer quality. To address these issues, we first construct a million-scale pose dataset across hundreds of distinct characters. We further propose MimiCAT, a cascade-transformer model designed for category-free 3D pose transfer. Instead of relying on strict one-to-one correspondence mappings, MimiCAT leverages semantic keypoint labels to learn a novel soft correspondence that enables flexible many-to-many matching across characters. The pose transfer is then formulated as a conditional generation process, in which the source transformations are first projected onto the target through soft correspondence matching and subsequently refined using shape-conditioned representations. Extensive qualitative and quantitative experiments demonstrate that MimiCAT transfers plausible poses across different characters, significantly outperforming prior methods that are limited to narrow category transfer (e.g., humanoid-to-humanoid).


## 📦 PokeAnimDB

### 🔎 Overview

This repository contains the released version of **PokeAnimDB**, a motion dataset used in our CVPR 2026 paper.

We release character metadata and motion sequences in `.npz` format. The dataset is intended for research on 3D character animation, motion analysis, motion synthesis, and category-free pose transfer.

The released version contains **900+** characters and **32k+** motion clips.

![screenshot](assets/teaser.gif)


## ⬇️ Download Dataset

The dataset is hosted on [Hugging Face](https://huggingface.co/datasets/mimicat3d/PokeAnimDB).

Download and extract it with:

```bash
hf download mimicat3d/PokeAnimDB data.tar.gz \
  --repo-type dataset \
  --local-dir ./PokeAnimDB
  
cd PokeAnimDB
tar -xzf data.tar.gz
cd ..
```

After extraction, the dataset will contain a `data/` directory with one folder per character.

### 📁 Dataset Structure
The files are organized as follows:
```text
PokeAnimDB/
├── data/
│   ├── pm0001_00_00/
│   │   ├── pm0001_00_00.npz
│   │   └── motion/
│   │       ├── attack01_0.npz
│   │       ├── ...
│   │       └── walk01_29.npz
│   └── ...
└── README.md
```

Each character directory contains:

- `<character_id>.npz`: rest-pose character metadata
- `motion/*.npz`: motion sequences with pose parameters and global translations

### 🎬 Demo
Install the renderer dependencies first:

```bash
pip install numpy scipy opencv-python pyrender trimesh pillow
```

By default, `quick_demo.py` reads data from `./PokeAnimDB/data` and writes videos to `./example`.

Use `quick_demo.py` to generate animated results from a rest-pose character file and a motion file:

```bash
python quick_demo.py --sample pm0001_00_00 --motion attack01_0
```

The `--motion` value should be the motion file name without the `.npz` suffix.

Rendered videos are saved under `example/<character_id>/`.

## 📜 License

This dataset is released for **research and non-commercial use** only under the
Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).


## 📖 Citation

If you find our work or dataset useful for your work, please cite:

```bibtex
@inproceedings{chai2026mimicat,
  author    = {Chai, Zenghao and Tang, Chen and Wong, Yongkang and Yang, Xulei and Kankanhalli, Mohan},
  title     = {MimiCAT: Mimic with Correspondence-Aware Cascade-Transformer for Category-Free 3D Pose Transfer},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
