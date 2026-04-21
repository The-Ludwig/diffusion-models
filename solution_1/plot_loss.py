from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torchvision as tv


def load_scalar_series(log_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    if not isinstance(log_dir, Path):
        log_dir = Path(log_dir)

    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    accumulator = EventAccumulator(str(log_dir), size_guidance={"images": 0})
    accumulator.Reload()

    available_tags = accumulator.Tags().get("scalars", [])
    if tag not in available_tags:
        raise ValueError(
            f"Tag '{tag}' not found in {log_dir}. Available scalar tags: {available_tags}"
        )

    events = accumulator.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    return np.array(steps), np.array(values)

def load_image_series(log_dir: Path, tag: str) -> list[np.ndarray]:
    if not isinstance(log_dir, Path):
        log_dir = Path(log_dir)

    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    accumulator = EventAccumulator(str(log_dir), size_guidance={"images": 0})
    accumulator.Reload()

    available_tags = accumulator.Tags().get("images", [])
    if tag not in available_tags:
        raise ValueError(
            f"Tag '{tag}' not found in {log_dir}. Available image tags: {available_tags}"
        )

    events = accumulator.Images(tag)
    images = [event.encoded_image_string for event in events]
    steps = [event.step for event in events]

    return steps, images

step, loss_train = load_scalar_series("runs/ddpm_mnist/train", "Loss/epoch")
step_val, loss_val = load_scalar_series("runs/ddpm_mnist/val", "Loss/epoch")
step_val_images, val_images = load_image_series("runs/ddpm_mnist/train", "Generated_Digits")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(step, loss_train, label="Train Loss", marker="x")
ax.plot(step_val, loss_val, label="Validation Loss", marker="x")

def smooth(x, w=11):
    s = np.r_[x[w-1:0:-1], x, x[-2:-w-1:-1]]
    return np.convolve(np.exp(-0.5*((np.arange(w)-w//2)/(w/6))**2), s, mode='valid')[w//2:-(w//2)] / np.exp(-0.5*((np.arange(w)-w//2)/(w/6))**2).sum()

# smooth validation loss and plot it in slight alpha
loss_smoothed = smooth(loss_val)
ax.plot(step_val, loss_smoothed, label="Validation Loss (smoothed)", color="C1", alpha=0.5)

ax.set_xlabel("Epoch")
ax.set_ylabel(r"Loss $L$")
ax.legend()

fig.savefig("solution_1/loss_plot.png")

output_dir = Path("solution_1/generated")
output_dir.mkdir(parents=True, exist_ok=True)

for image_step, encoded_image in zip(step_val_images, val_images):
    encoded_tensor = torch.tensor(list(encoded_image), dtype=torch.uint8)
    decoded_image = tv.io.decode_image(encoded_tensor)
    tv.io.write_png(decoded_image, str(output_dir / f"epoch_{image_step:03d}.png"))
