"""
Produce the guidance sweep plot

Usage:
    python sample.py                     # one sample per class at w=3
    python sample.py --w 7.0             # change guidance scale
    python sample.py --num-per-class 10  # more samples per class

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from model import MicroDiT, NUM_CLASSES, PAD_SIZE

from sample import sample_images, plot_sweep_grid, load_model

T = 1000
beta_start = 1e-4
beta_end = 0.02


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="micro_dit_checkpoint.pt")
    parser.add_argument("--w", type=float, nargs="*", default=[0.0, 1.0, 2.0, 4.0, 7.0, 10.0], help="guidance scales")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="sweep.png")
    parser.add_argument("--num-per-class", type=int, default=1)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_id)
        print(f"Using CUDA device: {gpu_name}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)

    N = args.num_per_class * NUM_CLASSES

    initial_noise = torch.randn(
        N, 1, PAD_SIZE, PAD_SIZE, device=device
    )

    model = load_model(args.checkpoint, device)
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(args.num_per_class)

    samples = {}
    pbar = tqdm(args.w)
    for w in pbar:
        pbar.set_postfix(weight=w)
        samples[w] = sample_images(model, labels, guidance_scale=w, device=device, initial_noise=initial_noise)

    pbar.close()

    plot_sweep_grid(samples, args.out)

    print(f"Saved {args.out}")
