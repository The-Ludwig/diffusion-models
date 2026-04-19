# Project 2: Deconstructing the Diffusion Transformer (MNIST)

This directory contains the code scaffolding for **Project 2**. A pre-trained
Micro-DiT checkpoint is shipped; you will *not* train a model. Your job is
to complete three small pieces of code and run the analyses described in the
project handout.

## Setup

Dependencies are managed at the repo root (`pyproject.toml`). From the root
of the repository:

```bash
pip install .
```

Run all commands **from the repository root** (the directory containing
`pyproject.toml`). Each command below launches the script as a module so
that relative paths resolve correctly:

```bash
cd path/to/assignment_repo
python -m assignment2.sample --w 3.0
```

If you prefer, you can also `cd src/assignment2 && python sample.py --w 3.0` --
both work, because the script loads the checkpoint from the same directory
it lives in and caches MNIST under `src/assignment2/data/`.

## Layout

```
src/assignment2/
├── model.py                 # MicroDiT (Task 1: implement AdaLayerNorm)
├── sample.py                # DDPM + CFG sampling loop (Task 3: two TODOs)
├── memorization.py          # pixel-space kNN + Improved PR (Tasks 4 and 5)
├── micro_dit_checkpoint.pt  # pre-trained weights (provided)
└── data/                    # MNIST download cache (auto-created)
```

## What you implement

| Task | File | What to do |
| ---- | ---- | ---------- |
| 1 | `model.py` | Fill in `AdaLayerNorm.__init__` and `AdaLayerNorm.forward`. |
| 3 | `sample.py` | Fill in the two `TODO` blocks inside `sample_images`: the CFG combination, and the DDPM posterior mean. |
| 4, 5 | new scripts of your choice | Use `memorization.pixel_l2_nearest_neighbor` (Task 4) and `memorization.improved_pr_pixel` (Task 5). |

After Task 1 your `model.py` should import without `NotImplementedError`.
After Task 3 you can run `python sample.py --w 3.0` to produce a single row
of one-per-class samples.

## Running the sampler

**All commands below assume you are in the `src/assignment2/` directory** so
that `micro_dit_checkpoint.pt` and the MNIST `data/` cache are next to the
script:

```bash
cd src/assignment2
python sample.py --w 3.0                     # w = 3, one sample per class
python sample.py --w 7.0 --num-per-class 10  # 10 samples per class at w = 7
python sample.py --w 0 --out samples_w0.png  # choose output filename
```

The script loads `micro_dit_checkpoint.pt` from the current directory by
default; pass `--checkpoint PATH` to override.

For Task 4, call `sample.sample_images` and `sample.plot_sweep_grid` from
your own script:

```python
import torch
from sample import load_model, sample_images, plot_sweep_grid, PAD_SIZE
from model import NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = load_model("micro_dit_checkpoint.pt", device)

labels = torch.arange(NUM_CLASSES, device=device)
torch.manual_seed(0)
noise  = torch.randn(NUM_CLASSES, 1, PAD_SIZE, PAD_SIZE, device=device)

rows = {}
for w in [0.0, 1.0, 2.0, 4.0, 7.0, 10.0]:
    rows[w] = sample_images(model, labels, guidance_scale=w,
                            initial_noise=noise.clone(), device=device)
plot_sweep_grid(rows, "task4_sweep.png")
```

## Loading MNIST in the same space as the samples

`memorization.load_mnist_tensors(train=True, limit=10_000)` returns MNIST
images as `(N, 1, 32, 32)` float tensors in `[-1, 1]` -- the same layout as
the sampler output, so you can feed generated and real tensors directly into
`improved_pr_pixel` or `pixel_l2_nearest_neighbor`.

## Hand-in

Submit a single PDF named `firstname_lastname_project_2.pdf` with your answers
to all five tasks. See the project handout for details.

If you get stuck, email the TA:
[taariq.nazar@math.su.se](mailto:taariq.nazar@math.su.se).
