import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from sklearn.metrics import confusion_matrix

keras.utils.set_random_seed(8)
VOCAB, LEN = 8000, 250
(xtr, ytr), (xte, yte) = keras.datasets.imdb.load_data(num_words=VOCAB)
xtr, xte = keras.utils.pad_sequences(xtr, maxlen=LEN), keras.utils.pad_sequences(xte, maxlen=LEN)
lengths = [len(s) for s in keras.datasets.imdb.load_data(num_words=VOCAB)[0][0]]
print(f"{len(xtr)} training reviews, {len(xte)} test reviews; average review {np.mean(lengths):.0f} words, cut or padded to {LEN}")


def build(kind):
    cell = {"SimpleRNN": layers.SimpleRNN(32), "LSTM": layers.LSTM(32), "GRU": layers.GRU(32), "BiGRU": layers.Bidirectional(layers.GRU(24))}[kind]
    return keras.Sequential([keras.layers.Input((LEN,)), layers.Embedding(VOCAB, 32, mask_zero=True), cell,
                             layers.Dropout(0.3), layers.Dense(1, activation="sigmoid")])


rows = {}
for kind in ("SimpleRNN", "LSTM", "GRU", "BiGRU"):
    m = build(kind)
    m.compile(optimizer=keras.optimizers.Adam(2e-3, clipnorm=1.0), loss="binary_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    h = m.fit(xtr, ytr, epochs=4, batch_size=128, validation_split=0.15, verbose=0)
    acc = m.evaluate(xte, yte, verbose=0)[1]
    rows[kind] = (m.count_params(), h.history["val_accuracy"], acc, time.time() - t0, m)
    print(f"{kind:<10} parameters {m.count_params():>6}  validation accuracy by epoch {[round(v, 3) for v in h.history['val_accuracy']]}  test accuracy {acc * 100:.2f} %  time {time.time() - t0:.0f} s")

best = max(rows, key=lambda k: rows[k][2])
print("\nbest model on the test reviews:", best)
pred = (rows[best][4].predict(xte, verbose=0).ravel() > 0.5).astype(int)
cm = confusion_matrix(yte, pred)
print("confusion matrix [[true negative, false positive], [false negative, true positive]]")
print(cm)
print(f"precision {cm[1, 1] / cm[:, 1].sum():.3f}  recall {cm[1, 1] / cm[1].sum():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for kind, st in zip(rows, ("-", "--", ":", "-.")):
    ax[0].plot(range(1, 5), rows[kind][1], st, marker="o", color="black", label=kind)
ax[0].set_title("Validation accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8)
ax[1].bar(list(rows), [rows[k][2] * 100 for k in rows], color="grey", edgecolor="black")
ax[1].set_ylim(40, 95); ax[1].set_title("Test accuracy (%)")
plt.tight_layout()
plt.show()
