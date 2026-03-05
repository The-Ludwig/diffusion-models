import argparse

import torch
import numpy as np

from assignment1.data import get_dataloader

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

from assignment1.sample import get_sample, load_model
from assignment1.classifier.helper import load_classifier

def inception_score(sample, classifier, splits=1):
    with torch.no_grad():
        p_pred = torch.softmax(classifier(sample), dim=-1)

    # Bootstrap-like estimate of variance
    split_len = len(sample) // splits
    scores = []
    for i in range(splits):
        part = p_pred[i*split_len:(i+1)*split_len]

        # this finds the mean in each class
        p_marginal = part.mean(dim=0) 
        
        # this sums over the different classes
        kl_div = (part*(part.log()-p_marginal.log())).sum(dim=1)
        
        scores.append(kl_div.mean().exp().item())

    return np.mean(scores), np.std(scores)
    

def get_gaussian_params(data):
    mean = data.mean(dim=0)
    cov = torch.cov(data.T)
    return mean, cov


def sqrtm(cov):
    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)
    
    # Handle small eigenvalues for numerical stability
    eigvals_clipped = torch.clamp(eigvals, min=1e-10)
    
    # Square root of eigenvalues
    sqrt_eigvals = torch.sqrt(eigvals_clipped)
    
    # Reconstruct the square root matrix
    sqrt_cov = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    return sqrt_cov
    

def fid(sample, real_sample, classifier, remove_last_layer=True):
    if remove_last_layer:
        # extract one layer above
        p_gen = classifier(sample, return_features=True)
        p_real = classifier(real_sample, return_features=True)
    else:
        p_gen = classifier(sample)
        p_real = classifier(real_sample)


    mean_gen, cov_gen = get_gaussian_params(p_gen)
    mean_real, cov_real = get_gaussian_params(p_real)

    # Frechet distance
    diff = mean_gen - mean_real
    covmean = sqrtm(cov_gen @ cov_real)
    
    return (diff @ diff + torch.trace(cov_gen + cov_real - 2 * covmean)).item()


def main(n_samples=100):
    classifier = load_classifier(device="cpu")
    model = load_model()

    with torch.no_grad():
        sample = get_sample(model, n_samples=n_samples).to("cpu")

    print(f"Inception score: {inception_score(sample, classifier, splits=10)}")
    print(f"Inception score: {inception_score(sample, classifier, splits=1)}")


    # Double check
    score = InceptionScore(feature=classifier, normalize=False, compute_with_cache=False)
    
    print(f"Inception score (torchmetrics): {score(sample)}")

    # FID 
    # Load the real samples from MNIST 
    dl = get_dataloader(batch_size=n_samples)
    real_sample, _ = next(iter(dl))
    real_sample = real_sample.to(sample.device)

    print(f"FID: {fid(sample, real_sample, classifier)}")
    print(f"FID (before softmax): {fid(sample, real_sample, classifier, remove_last_layer=False)}")

    fid_score = FrechetInceptionDistance(
        feature=classifier,
        input_img_size=real_sample.shape[-3:],
        normalize=True,
    ).to(sample.device)
    fid_score.update(sample, real=False)

    fid_score.update(real_sample, real=True)

    print(f"FID (torchmetrics): {fid_score.compute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1_000)
    args = parser.parse_args()

    main(n_samples=args.n_samples)
