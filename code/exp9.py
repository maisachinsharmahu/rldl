import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

keras.utils.set_random_seed(9)
cats = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.guns"]
strip = ("headers", "footers", "quotes")
tr = fetch_20newsgroups(subset="train", categories=cats, remove=strip)
te = fetch_20newsgroups(subset="test", categories=cats, remove=strip)
print("topics:", cats)
print("training posts:", len(tr.data), " test posts:", len(te.data), " average length:", int(np.mean([len(d.split()) for d in tr.data])), "words")

t0 = time.time()
tfidf = TfidfVectorizer(max_features=20000, stop_words="english")
base = LogisticRegression(max_iter=1000).fit(tfidf.fit_transform(tr.data), tr.target)
base_pred = base.predict(tfidf.transform(te.data))
base_acc = accuracy_score(te.target, base_pred)
print(f"\nbaseline TF-IDF + logistic regression: test accuracy {base_acc * 100:.2f} %  ({time.time() - t0:.1f} s)")

vec = layers.TextVectorization(max_tokens=15000, output_sequence_length=200)
vec.adapt(np.array(tr.data))
Xtr, Xte = vec(np.array(tr.data)).numpy(), vec(np.array(te.data)).numpy()
model = keras.Sequential([
    keras.layers.Input((200,)),
    layers.Embedding(15000, 64, mask_zero=True),
    layers.Bidirectional(layers.LSTM(48, return_sequences=True)),
    layers.GlobalMaxPooling1D(),
    layers.Dropout(0.4),
    layers.Dense(len(cats), activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
t0 = time.time()
hist = model.fit(Xtr, tr.target, epochs=5, batch_size=64, validation_split=0.1, verbose=0)
pred = model.predict(Xte, verbose=0).argmax(axis=1)
acc = accuracy_score(te.target, pred)
print(f"embedding + bidirectional LSTM ({model.count_params()} parameters): test accuracy {acc * 100:.2f} %  ({time.time() - t0:.0f} s)")
for e, (a, v) in enumerate(zip(hist.history["accuracy"], hist.history["val_accuracy"]), 1):
    print(f"  epoch {e}: train accuracy {a:.3f}  validation accuracy {v:.3f}")

emb = model.layers[0].get_weights()[0]
vocab = vec.get_vocabulary()
index = {w: i for i, w in enumerate(vocab)}


def neighbours(word, k=4):
    v = emb[index[word]]
    sims = emb @ v / (np.linalg.norm(emb, axis=1) * np.linalg.norm(v) + 1e-9)
    return [str(vocab[i]) for i in np.argsort(-sims)[1:k + 1]]


print("\nwords closest to some topic words in the learned embedding")
for w in ("nasa", "hockey", "graphics", "gun"):
    if w in index:
        print(f"  {w:<9}->", neighbours(w))

cm = confusion_matrix(te.target, pred)
print("\nconfusion matrix of the LSTM model (rows = true topic)")
print(cm)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(["TF-IDF +\nlogistic", "BiLSTM"], [base_acc * 100, acc * 100], color=["lightgrey", "grey"], edgecolor="black")
ax[0].set_ylim(50, 100); ax[0].set_title("Test accuracy (%)")
ax[1].imshow(cm, cmap="Greys")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels([c.split(".")[-1] for c in cats], rotation=30, fontsize=8)
ax[1].set_yticks(range(4)); ax[1].set_yticklabels([c.split(".")[-1] for c in cats], fontsize=8)
for i in range(4):
    for j in range(4):
        ax[1].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax[1].set_title("BiLSTM confusion matrix")
plt.tight_layout()
plt.show()
