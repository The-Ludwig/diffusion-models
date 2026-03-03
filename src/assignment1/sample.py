import torch
from tqdm import tqdm
from torchvision import utils as tvu
from pathlib import Path
from assignment1.model import UNet

SAVE_STEPS = [300, 250, 200, 150, 100, 50, 0]

@torch.no_grad()
def sample(
    model, n_samples, image_channels, image_size, T, alpha, alpha_cumprod, beta, device, 
    save_t=None, save_path=None
):
    model.eval()
    # Start from pure noise
    x = torch.randn((n_samples, image_channels, image_size, image_size), device=device)

    if save_t is not None and save_path is not None:
        if T in save_t:
            tvu.save_image(x, save_path/f"{T:03d}.png")


    for i in tqdm(reversed(range(T)), total=T, desc="Sampling", leave=False):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x, t)

        # Get coefficients for the current timestep
        alpha_t = alpha[i]
        alpha_cumprod_t = alpha_cumprod[i]
        beta_t = beta[i]

        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        # DDPM Reverse Step formula
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t) * eps_theta) + sigma_t * noise
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        )
        std = torch.sqrt(beta_t)

        x = mean + std * noise

        model.train()

        if save_t is not None and save_path is not None:
            if i in save_t:
                tvu.save_image(x, save_path/f"{i:03d}.png")

    # Rescale from [-1, 1] back to [0, 1] for visualization
    x = (x.clamp(-1, 1) + 1) / 2
    return x


if __name__ == "__main__":
    # Hyperparams (assume stays as trained)
    T = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DDPM Schedule (assume stays as trained)
    beta = torch.linspace(1e-4, 0.02, T).to(device)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    # Load latest model 
    model = UNet().to(device) 
    model_weights = torch.load("ddpm_mnist_final.pth")

    model.load_state_dict(model_weights['model'], strict=True)

    # generate images
    path = Path("solution_1/generated/")
    path.mkdir(exist_ok=True, parents=True)
    sample(model, 5, 1, 28, T, alpha, alpha_cumprod, beta, device, save_t=SAVE_STEPS, save_path=path)

