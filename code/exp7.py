import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(7)
(xtr, ytr), (xte, yte) = keras.datasets.mnist.load_data()
Xtr = torch.tensor(xtr / 255.0, dtype=torch.float32).unsqueeze(1)
Xte = torch.tensor(xte / 255.0, dtype=torch.float32).unsqueeze(1)
Ytr, Yte = torch.tensor(ytr).long(), torch.tensor(yte).long()
print("training set", tuple(Xtr.shape), " test set", tuple(Xte.shape))

conv = nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2))
head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 5 * 5, 64), nn.ReLU(), nn.Dropout(0.25), nn.Linear(64, 10))
model = nn.Sequential(conv, head)
print("feature extractor output:", tuple(conv(Xtr[:1]).shape[1:]), "-> flattened 400 -> dense 64 -> 10 classes")
print("trainable parameters:", sum(p.numel() for p in model.parameters()))

opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
t0 = time.time()
curve = []
for ep in range(1, 5):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 128):
        b = perm[i:i + 128]
        loss = nn.functional.cross_entropy(model(Xtr[b]), Ytr[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    sched.step()
    model.eval()
    with torch.no_grad():
        acc = (torch.cat([model(Xte[i:i + 1000]).argmax(1) for i in range(0, 10000, 1000)]) == Yte).float().mean().item()
    curve.append(acc)
    print(f"epoch {ep}: learning rate {opt.param_groups[0]['lr']:.4f}, last loss {loss.item():.4f}, test accuracy {acc * 100:.2f} %")
print(f"training time {time.time() - t0:.0f} s")

with torch.no_grad():
    pred = torch.cat([model(Xte[i:i + 1000]).argmax(1) for i in range(0, 10000, 1000)])
per_digit = [(pred[Yte == d] == d).float().mean().item() * 100 for d in range(10)]
print("accuracy per digit:", " ".join(f"{d}:{a:.1f}" for d, a in enumerate(per_digit)))
print("total mistakes:", int((pred != Yte).sum()), "of 10000")

sample = Xte[0:1]
with torch.no_grad():
    maps = conv[:3](sample)[0]
fig, ax = plt.subplots(2, 5, figsize=(12, 5))
ax[0, 0].imshow(sample[0, 0], cmap="Greys"); ax[0, 0].set_title(f"input (digit {yte[0]})")
filters = conv[0].weight.detach()[:, 0]
for k in range(4):
    ax[0, k + 1].imshow(filters[k], cmap="Greys"); ax[0, k + 1].set_title(f"filter {k + 1}")
for k in range(5):
    ax[1, k].imshow(maps[k], cmap="Greys"); ax[1, k].set_title(f"feature map {k + 1}")
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
plt.show()
