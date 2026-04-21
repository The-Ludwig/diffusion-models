import argparse

import torch
from tqdm import tqdm
from torchvision import utils as tvu
from pathlib import Path
from assignment1.classifier.helper import load_classifier
from assignment1.ema import EMA
from assignment1.model import UNet

SAVE_STEPS = [300, 250, 200, 150, 100, 50, 0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# @torch.no_grad()
def sample(
    model, n_samples, image_channels, image_size, T, alpha, alpha_cumprod, beta, device, 
    save_t=None, save_path=None,
    guidance=None, guidance_scale=.7,
):

    if guidance is not None:
        guidance = torch.full((n_samples,), guidance, device=device, dtype=torch.long)

    model.eval()
    # Start from pure noise
    x = torch.randn((n_samples, image_channels, image_size, image_size), device=device)

    if save_t is not None and save_path is not None:
        if T in save_t:
            tvu.save_image(x, save_path/f"{T:03d}.png")


    for i in tqdm(reversed(range(T)), total=T, desc="Sampling", leave=False):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)

        # Predict noise
        if guidance is not None and guidance_scale > 0:
            predicted_noise = model(x, t)
            predicted_noise_guided = model(x, t, guidance)
            predicted_noise = predicted_noise + guidance_scale * (predicted_noise_guided - predicted_noise)
        else:
            predicted_noise = model(x, t)

        alpha_t = alpha[i]
        alpha_cumprod_t = alpha_cumprod[i]
        beta_t = beta[i]

        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        # Get coefficients for the current timestep
        # DDPM Reverse Step formula
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t) * eps_theta) + sigma_t * noise
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        )
        std = torch.sqrt(beta_t)

        x = mean + std * noise


        if save_t is not None and save_path is not None:
            if i in save_t:
                tvu.save_image(x, save_path/f"{i:03d}.png")

    model.train()

    # Rescale from [-1, 1] back to [0, 1] for visualization
    x = (x.clamp(-1, 1) + 1) / 2
    return x


def load_model():
    # Hyperparams (assume stays as trained)
    T = 300

    # DDPM Schedule (assume stays as trained)
    beta = torch.linspace(1e-4, 0.02, T).to(DEVICE)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    # Load latest model 
    model = UNet(n_classes=10).to(DEVICE) 
    model_weights = torch.load("ddpm_mnist_final.pth")

    ema = EMA(model, decay=0.9999)
    ema.shadow = model_weights['ema']
    ema.apply_shadow()

    model.load_state_dict(model_weights['model'], strict=True)
    return model

def get_sample(model, n_samples=1):
    T = 300

    # DDPM Schedule (assume stays as trained)
    beta = torch.linspace(1e-4, 0.02, T).to(DEVICE)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    s = sample(model, n_samples, 1, 28, T, alpha, alpha_cumprod, beta, DEVICE)
    # rescale from [0, 1] to [-1, 1]
    return s*2 - 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--guidance", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    args = parser.parse_args()

    model = load_model()

    T = 300
    beta = torch.linspace(1e-4, 0.02, T).to(DEVICE)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    # generate images
    path = Path("solution_1/generated/")
    path.mkdir(exist_ok=True, parents=True)

    sample(model, 5, 1, 28, T, alpha, alpha_cumprod, beta, DEVICE, save_t=SAVE_STEPS, save_path=path, guidance=args.guidance, guidance_scale=args.guidance_scale)

