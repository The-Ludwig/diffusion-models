# Deep Generative Modeling — Assignments

Single repository for all assignment code in the course. Each project lives
under its own sub-package of `src/` and is self-contained.

## Layout

- `src/assignment1/` — Project 1: DDPM on MNIST (U-Net, training, FID/IS).
  See [`src/assignment1/README.md`](src/assignment1/README.md).
- `src/assignment2/` — Project 2: Micro-DiT on MNIST (AdaLN, CFG, memorization
  analysis). See [`src/assignment2/README.md`](src/assignment2/README.md).

## Setup

Dependencies are managed at the repo root via `pyproject.toml`
(Python 3.9+):

```bash
pip install .
```

Pre-trained checkpoints shipped with the repo:

- `ddpm_mnist_final.pth` — Project 1 (repo root).
- `src/assignment2/micro_dit_checkpoint.pt` — Project 2.
