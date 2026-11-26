import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from importlib import metadata

import keras
import numpy as np
import tensorflow as tf
import torch

print("versions: tensorflow", metadata.version("tensorflow"), "| keras", metadata.version("keras"), "| torch", metadata.version("torch"))
print("TensorFlow devices:", [d.device_type for d in tf.config.list_physical_devices()])
print("PyTorch: cuda", torch.cuda.is_available(), "| mps", torch.backends.mps.is_available(), "| threads", torch.get_num_threads())

a = np.arange(6, dtype="float32").reshape(2, 3)
b = np.ones((3, 2), dtype="float32")
print("\nmatrix product in three libraries (2x3 times 3x2)")
print("  numpy     :", (a @ b).tolist())
print("  tensorflow:", tf.matmul(a, b).numpy().tolist())
print("  pytorch   :", (torch.tensor(a) @ torch.tensor(b)).tolist())
print("broadcasting a row onto a column:", (torch.arange(3).view(3, 1) + torch.arange(4)).shape)

rng = np.random.default_rng(1)
x = rng.uniform(-2, 2, 200).astype("float32")
y = (3 * x + 2 + rng.normal(0, 0.2, 200)).astype("float32")
print("\nlearn y = 3x + 2 from 200 noisy points (3 libraries)")

w, c = tf.Variable(0.0), tf.Variable(0.0)
for _ in range(300):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((w * x + c - y) ** 2)
    gw, gc = tape.gradient(loss, [w, c])
    w.assign_sub(0.1 * gw)
    c.assign_sub(0.1 * gc)
print(f"  TensorFlow GradientTape : slope {float(w):.3f}  intercept {float(c):.3f}")

keras.utils.set_random_seed(1)
m = keras.Sequential([keras.layers.Input((1,)), keras.layers.Dense(1)])
m.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")
m.fit(x, y, epochs=60, batch_size=32, verbose=0)
kw, kb = m.layers[0].get_weights()
print(f"  Keras Dense layer       : slope {kw[0][0]:.3f}  intercept {kb[0]:.3f}")

lin = torch.nn.Linear(1, 1)
opt = torch.optim.SGD(lin.parameters(), lr=0.1)
xt, yt = torch.tensor(x).view(-1, 1), torch.tensor(y).view(-1, 1)
for _ in range(300):
    opt.zero_grad()
    torch.nn.functional.mse_loss(lin(xt), yt).backward()
    opt.step()
print(f"  PyTorch nn.Linear       : slope {lin.weight.item():.3f}  intercept {lin.bias.item():.3f}")
print("\nnumber of parameters: Keras", m.count_params(), "| PyTorch", sum(p.numel() for p in lin.parameters()))
