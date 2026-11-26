# Experiment 2: Installing and Using Keras, TensorFlow and PyTorch

> **File:** `exp2.py`  |  **Tools:** Python 3, TensorFlow, Keras, PyTorch, NumPy

## 🎯 Aim

To install the Keras, TensorFlow and PyTorch libraries and to make use of them.

## 📘 Theory

TensorFlow (Google) and PyTorch (Meta) are the two leading deep learning frameworks. Both are built on tensors, which are n-dimensional arrays that can live on a CPU or on an accelerator, and both compute gradients automatically, which is what makes training possible. TensorFlow records the operations on a tf.GradientTape, while PyTorch builds a graph dynamically and calls backward(). Keras is the high-level API in which a model is assembled from layers; Keras 3 can run on TensorFlow, PyTorch or JAX.

The libraries are installed with pip install tensorflow keras torch. After the installation the version, the available devices (CPU, a CUDA GPU or Apple's MPS) and a few tensor operations are checked. Then the same problem, learning y = 3x + 2 from 200 noisy points, is solved three times: with a manual gradient-descent loop on a TensorFlow GradientTape, with a Keras Dense layer, and with a PyTorch nn.Linear module. All three should find a slope near 3 and an intercept near 2.

- **Tensor**: N-dimensional array that can be moved to an accelerator.
- **GradientTape / autograd**: Mechanisms that record operations and compute gradients.
- **Broadcasting**: Automatic expansion of smaller arrays to match larger ones.
- **MPS / CUDA**: Apple and NVIDIA back-ends that run tensor operations on a GPU.

## 🪜 Algorithm / Steps

1. Install the three libraries with pip install tensorflow keras torch.
2. Print their versions and the devices each of them can use.
3. Multiply a 2 × 3 and a 3 × 2 matrix in NumPy, TensorFlow and PyTorch and show broadcasting.
4. Generate 200 noisy points of y = 3x + 2 and fit them with a GradientTape loop, with a Keras Dense layer and with nn.Linear.
5. Compare the learned slope and intercept and the number of parameters.

## 💻 Code

```python
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
```

## 🖥️ Output

```text
versions: tensorflow 2.21.0 | keras 3.15.1 | torch 2.15.0
TensorFlow devices: ['CPU']
PyTorch: cuda False | mps True | threads 4

matrix product in three libraries (2x3 times 3x2)
  numpy     : [[3.0, 3.0], [12.0, 12.0]]
  tensorflow: [[3.0, 3.0], [12.0, 12.0]]
  pytorch   : [[3.0, 3.0], [12.0, 12.0]]
broadcasting a row onto a column: torch.Size([3, 4])

learn y = 3x + 2 from 200 noisy points (3 libraries)
  TensorFlow GradientTape : slope 2.993  intercept 1.983
  Keras Dense layer       : slope 2.978  intercept 1.968
  PyTorch nn.Linear       : slope 2.993  intercept 1.983

number of parameters: Keras 2 | PyTorch 2
```

## ✅ Result

TensorFlow 2.21.0, Keras 3.15.1 and PyTorch 2.15.0 are installed. TensorFlow sees only the CPU, while PyTorch reports that the Apple MPS back-end is available. All three libraries give the same matrix product [[3, 3], [12, 12]]. The fits are close to the true line: slope 2.993 and intercept 1.983 for GradientTape and for PyTorch, and 2.978 and 1.968 for Keras (60 epochs, mini-batches). Each model has just 2 parameters.

## 🎤 Viva Questions

**Q. How are the libraries installed?**  
A. With pip install tensorflow keras torch.

**Q. What is a tensor?**  
A. An n-dimensional array that can be used on a CPU or a GPU.

**Q. How are gradients computed automatically?**  
A. TensorFlow records operations on a GradientTape and PyTorch uses autograd with backward().

**Q. What is Keras?**  
A. A high-level API that builds neural networks from layers; Keras 3 runs on TensorFlow, PyTorch or JAX.

**Q. What is broadcasting?**  
A. Automatic expansion of a smaller array to the shape of a larger one in an operation.

---
[⬅ Back to index](README.md)
