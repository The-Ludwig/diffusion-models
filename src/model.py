import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.relu(self.conv1(x))
        # Add time conditioning
        time_emb = self.relu(self.time_mlp(t))[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.relu(self.conv2(h))
        return h


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        t_dim = 32
        self.time_mlp = nn.Sequential(
            TimeEmbedding(t_dim), nn.Linear(t_dim, t_dim), nn.ReLU()
        )

        # Extremely small channel progression for speed: 1 -> 32 -> 64
        self.down1 = Block(1, 32, t_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(32, 64, t_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = Block(64, 64, t_dim)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up_block1 = Block(128, 32, t_dim)  # 64 (up) + 64 (skip)
        self.up2 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.up_block2 = Block(64, 1, t_dim)  # 32 (up) + 32 (skip)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x1 = self.down1(x, t)
        x2 = self.down2(self.pool1(x1), t)

        out = self.bottleneck(self.pool2(x2), t)

        out = self.up1(out)
        out = torch.cat([out, x2], dim=1)
        out = self.up_block1(out, t)

        out = self.up2(out)
        # Pad slightly if MNIST 28x28 doesn't align perfectly with power-of-2 strides
        if out.shape != x1.shape:
            out = nn.functional.interpolate(out, size=(28, 28))

        out = torch.cat([out, x1], dim=1)
        return self.up_block2(out, t)
