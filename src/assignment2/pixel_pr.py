"""
Memorization analysis helpers for Project 2 (Tasks 4 and 5).

All distances here are computed directly in **pixel space** on flattened
(1, 32, 32) -> 1024-dim tensors. No classifier features are used.

Provided:
  - pixel_l2_nearest_neighbor: L2 nearest neighbor of a generated image in
    a set of training images (used for Task 4).
  - improved_pr_pixel: Improved Precision and Recall with k=5 in pixel
    space (used for Task 5).
  - load_mnist_tensors: convenience loader that returns MNIST images in the
    same (N, 1, 32, 32) layout and [-1, 1] range as the sampler output.
"""
from __future__ import annotations

import argparse
from tqdm import tqdm
import numpy as np

from model import MicroDiT, NUM_CLASSES, PAD_SIZE
from sample import sample_images, plot_sweep_grid, load_model
from memorization import load_mnist_tensors

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

DATA_DIR = Path(__file__).absolute().parents[2]/"data"

@torch.no_grad()
def improved_pr_pixel(generated: torch.Tensor, real: torch.Tensor, k: int = 5):
    """Improved Precision and Recall in pixel space (Kynkaanniemi et al., 2019).

    For each point the method defines a hypersphere whose radius equals the
    distance to its k-th nearest neighbor *within its own set*. A generated
    sample is counted as "precise" if it falls inside at least one real
    hypersphere; a real sample is "recalled" if it falls inside at least one
    generated hypersphere.

    Args:
        generated: (N_gen, C, H, W) generated images.
        real:      (N_real, C, H, W) real images.
        k: number of nearest neighbors used to define hypersphere radii.

    Returns:
        precision: float in [0, 1].
        recall:    float in [0, 1].
    """
    gen_flat = generated.reshape(generated.shape[0], -1).float()
    real_flat = real.reshape(real.shape[0], -1).float()

    real_dists = torch.cdist(real_flat, real_flat)
    real_radii = real_dists.topk(k + 1, dim=1, largest=False).values[:, -1]

    gen_dists = torch.cdist(gen_flat, gen_flat)
    gen_radii = gen_dists.topk(k + 1, dim=1, largest=False).values[:, -1]

    cross_dists = torch.cdist(gen_flat, real_flat)

    inside_real = (cross_dists < real_radii.unsqueeze(0)).any(dim=1)
    precision = float(inside_real.float().mean().item())

    inside_gen = (cross_dists.t() < gen_radii.unsqueeze(0)).any(dim=1)
    recall = float(inside_gen.float().mean().item())

    return precision, recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="micro_dit_checkpoint.pt")
    parser.add_argument("--w", type=float, nargs="*", default=[0, 1, 2, 4, 7, 10], help="guidance scales")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="pixel_pr.png")
    parser.add_argument("--num-per-class", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
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

    print("Loading test data")
    test_data, test_labels = load_mnist_tensors(train=False, limit=500)
    print("Done.")
    test_data, test_labels = test_data.to(device), test_labels.to(device)


    # Do the precision-recall on all classes combined
    precision, recall = {}, {}

    for w in args.w:
        p, r = improved_pr_pixel(samples[w], test_data, k=args.k)
        precision[w] = p
        recall[w] = r

    fig, ax = plt.subplots()
    
    ax.plot(list(precision.keys()), list(precision.values()), label="Precision", marker="x")
    ax.plot(list(recall.keys()), list(recall.values()), label="Recall", marker="x")

    ax.legend()
    ax.set_xlabel("Guicance Scale $w$")
    ax.set_ylabel("Precision/Recall")

    fig.savefig(args.out)
   
    print(f"Saved {args.out}")

    # Do the precision-recall per-class
    precision, recall = [{} for _ in range(NUM_CLASSES)], [{} for _ in range(NUM_CLASSES)]

    for w in args.w:
        for c in range(NUM_CLASSES):
            gen_mask = labels == c
            test_mask = test_labels == c

            p, r = improved_pr_pixel(samples[w][gen_mask], test_data[test_mask], k=args.k)
            precision[c][w] = p
            recall[c][w] = r

    overall_precision = {w: np.mean([precision[c][w] for c in range(NUM_CLASSES)]) for w in args.w}
    overall_recall = {w: np.mean([recall[c][w] for c in range(NUM_CLASSES)]) for w in args.w}

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(list(overall_precision.keys()), list(overall_precision.values()), label="Precision", marker="x")
    axs[0].plot(list(overall_recall.keys()), list(overall_recall.values()), label="Recall", marker="x")

    axs[0].legend()
    axs[0].set_xlabel("Guicance Scale $w$")
    axs[0].set_ylabel("Precision/Recall")

    cmap = plt.get_cmap('autumn')
    colors = cmap(np.linspace(0, 1, NUM_CLASSES))

    for c, (p, r, color) in enumerate(zip(precision, recall, colors)):
        axs[1].plot(list(precision[c].keys()), list(precision[c].values()), color=color, linestyle="dashed", label=f"Precision" if c==0 else None, marker="x")
        axs[1].plot(list(recall[c].keys()), list(recall[c].values()), color=color, linestyle="solid", label=f"Recall" if c==0 else None, marker="x")

    # make a colorbar 
    norm = Normalize(0, NUM_CLASSES)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axs[1], label="Class label")

    axs[1].legend()
    axs[1].set_xlabel("Guicance Scale $w$")
    axs[1].set_ylabel("Precision/Recall")

    fig.savefig(f"classwise_{args.out}")
   
    print(f"Saved classwise_{args.out}")
