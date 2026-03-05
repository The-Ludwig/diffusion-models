from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar_series(log_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    if not isinstance(log_dir, Path):
        log_dir = Path(log_dir)

    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    accumulator = EventAccumulator(str(log_dir))
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

step, loss_train = load_scalar_series("runs/ddpm_mnist/train", "Loss/epoch")
step_val, loss_val = load_scalar_series("runs/ddpm_mnist/val", "Loss/epoch")

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