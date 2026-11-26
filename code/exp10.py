import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

keras.utils.set_random_seed(10)
faces = fetch_olivetti_faces()
X = faces.images[..., None].astype("float32")
y = faces.target
print(f"Olivetti faces: {len(X)} images of {len(set(y))} people ({len(X) // len(set(y))} each), size {X.shape[1]} x {X.shape[2]}")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=4)
mean = Xtr.mean()
Xtr, Xte = Xtr - mean, Xte - mean
print("training images:", len(Xtr), " test images:", len(Xte), "(unseen photos of the same people)")

aug = keras.Sequential([layers.RandomTranslation(0.08, 0.08), layers.RandomRotation(0.06), layers.RandomZoom(0.1), layers.RandomContrast(0.15)])
inputs = keras.Input((64, 64, 1))
x = aug(inputs)
for f in (16, 32, 64):
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(40, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("parameters:", model.count_params())

hist = model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=100, batch_size=32, verbose=0)
for e in (9, 24, 49, 74, 99):
    print(f"epoch {e + 1:>2}: train accuracy {hist.history['accuracy'][e]:.3f}  test accuracy {hist.history['val_accuracy'][e]:.3f}")

probs = model.predict(Xte, verbose=0)
pred = probs.argmax(1)
acc = (pred == yte).mean()
print(f"\nfinal test accuracy {acc * 100:.2f} %  ({int((pred != yte).sum())} wrong out of {len(yte)})")
top5 = np.mean([yte[i] in np.argsort(probs[i])[::-1][:5] for i in range(len(yte))])
print(f"correct person among the top five guesses: {top5 * 100:.1f} %")
wrong = np.where(pred != yte)[0]
for i in wrong[:6]:
    print(f"  photo of person {yte[i]:>2} taken for person {pred[i]:>2} (confidence {probs[i].max():.2f})")

fig, ax = plt.subplots(2, 6, figsize=(12, 5.2))
ax = ax.ravel()
for k in range(6):
    ax[k].imshow(Xte[k, :, :, 0] + mean, cmap="gray")
    ax[k].set_title(f"true {yte[k]}, pred {pred[k]}", fontsize=8)
for k in range(5):
    ax[6 + k].imshow(model.layers[2].get_weights()[0][:, :, 0, k], cmap="gray")
    ax[6 + k].set_title(f"filter {k + 1}", fontsize=8)
ax[11].plot(hist.history["accuracy"], color="black", label="train")
ax[11].plot(hist.history["val_accuracy"], color="grey", label="test")
ax[11].set_title("accuracy", fontsize=8); ax[11].legend(fontsize=6)
for a in ax[:11]:
    a.axis("off")
plt.tight_layout()
plt.show()
