import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(31)
(xtr, ytr), _ = keras.datasets.mnist.load_data()
X = torch.tensor(xtr.reshape(-1, 784) / 127.5 - 1.0, dtype=torch.float32)
Y = torch.tensor(ytr).long()
print("training images:", tuple(X.shape), "| labels 0-9 are given to both networks (conditional GAN)")

ZDIM = 48


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label = nn.Embedding(10, 16)
        self.net = nn.Sequential(nn.Linear(ZDIM + 16, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784), nn.Tanh())

    def forward(self, z, y):
        return self.net(torch.cat([z, self.label(y)], dim=1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label = nn.Embedding(10, 16)
        self.net = nn.Sequential(nn.Linear(784 + 16, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x, y):
        return self.net(torch.cat([x, self.label(y)], dim=1))


G, D = Generator(), Discriminator()
og = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
od = torch.optim.Adam(D.parameters(), lr=8e-5, betas=(0.5, 0.999))
loss_fn = nn.BCEWithLogitsLoss()
print("parameters: generator", sum(p.numel() for p in G.parameters()), "| discriminator", sum(p.numel() for p in D.parameters()))

fixed_z = torch.randn(80, ZDIM)
fixed_y = torch.arange(10).repeat_interleave(8)
history, shots, t0 = [], {}, time.time()
for ep in range(1, 31):
    perm = torch.randperm(len(X))
    for i in range(0, len(perm) - 127, 128):
        idx = perm[i:i + 128]
        real, ry = X[idx], Y[idx]
        fy = torch.randint(0, 10, (len(idx),))
        fake = G(torch.randn(len(idx), ZDIM), fy)
        noise = 0.1 * max(0.0, 1 - ep / 20)
        ld = loss_fn(D(real + noise * torch.randn_like(real), ry), torch.full((len(idx), 1), 0.9)) + loss_fn(D(fake.detach() + noise * torch.randn_like(fake), fy), torch.zeros(len(idx), 1))
        od.zero_grad(); ld.backward(); od.step()
        lg = loss_fn(D(fake, fy), torch.ones(len(idx), 1))
        og.zero_grad(); lg.backward(); og.step()
    history.append((ld.item(), lg.item()))
    if ep in (1, 10, 30):
        with torch.no_grad():
            shots[ep] = G(fixed_z, fixed_y).view(10, 8, 28, 28).numpy()
    if ep % 5 == 0 or ep == 1:
        print(f"epoch {ep:>2}  discriminator loss {ld.item():.3f}  generator loss {lg.item():.3f}  ({time.time() - t0:.0f} s)")

with torch.no_grad():
    G.eval()
    D.eval()
    z = torch.randn(1000, ZDIM)
    y = torch.randint(0, 10, (1000,))
    fake_scores = torch.sigmoid(D(G(z, y), y)).mean().item()
    real_scores = torch.sigmoid(D(X[:1000], Y[:1000])).mean().item()
print(f"\ndiscriminator confidence that an image is real: real images {real_scores:.2f}, generated images {fake_scores:.2f}")

fig, ax = plt.subplots(1, 3, figsize=(14, 5))
for a, (ep, g) in zip(ax, shots.items()):
    a.imshow(np.vstack([np.hstack(g[r]) for r in range(10)]), cmap="Greys")
    a.set_title(f"epoch {ep}: each row is one digit 0 to 9"); a.axis("off")
plt.tight_layout()
plt.show()
