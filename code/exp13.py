import math
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

cats = ["rec.motorcycles", "sci.crypt", "talk.politics.mideast"]
raw = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
docs, label = [], []
for c in range(len(cats)):
    chosen = [d for d, t in zip(raw.data, raw.target) if t == c and 100 < len(d.split()) < 400][:8]
    docs += chosen
    label += [raw.target_names[c].split(".")[-1]] * len(chosen)
print(f"{len(docs)} documents, 8 for each of {raw.target_names}")

words = [re.findall(r"[a-z]+(?:'[a-z]+)?", d.lower()) for d in docs]
vocab = sorted({w for ws in words for w in ws})
index = {w: i for i, w in enumerate(vocab)}
N, V = len(docs), len(vocab)
print("vocabulary size:", V, "| tokens:", sum(map(len, words)))

counts = np.zeros((N, V))
for i, ws in enumerate(words):
    for w, c in Counter(ws).items():
        counts[i, index[w]] = c

stop = np.array([w in ENGLISH_STOP_WORDS for w in vocab])
stop_tokens = counts[:, stop].sum(axis=1)
print("\nshare of stop words in the text of each topic")
for c in sorted(set(label)):
    rows = [i for i in range(N) if label[i] == c]
    print(f"  {c:<12} {100 * stop_tokens[rows].sum() / counts[rows].sum():5.1f} %")
overall = counts[:, stop].sum(axis=0)
order = np.argsort(-overall)[:8]
print("\nfrequent stop words, corpus total and number of documents that contain them")
for j in order:
    print(f"  {np.array(vocab)[stop][j]:<6} {int(overall[j]):>4} times in {int((counts[:, stop][:, j] > 0).sum()):>2} of {N} documents")

tf = counts / counts.sum(axis=1, keepdims=True)
df = (counts > 0).sum(axis=0)
idf = np.log((1 + N) / (1 + df)) + 1
tfidf = tf * idf
tfidf /= np.linalg.norm(tfidf, axis=1, keepdims=True)
print("\nsmoothed idf = ln((1 + N) / (1 + df)) + 1")
for w in ("the", "and", "bike", "key", "israel"):
    print(f"  {w:<7} df = {int(df[index[w]]):>2}  idf = {idf[index[w]]:.3f}")

print("\ndocuments ranked by cosine similarity with the tf-idf vector of the query")
for q in ("engine ride motorcycle helmet", "encryption key algorithm", "israel arab peace war"):
    qv = np.zeros(V)
    for w in re.findall(r"[a-z]+(?:'[a-z]+)?", q):
        if w in index:
            qv[index[w]] = idf[index[w]] / len(q.split())
    qv /= np.linalg.norm(qv)
    sims = tfidf @ qv
    top = np.argsort(-sims)[:3]
    print(f"  '{q}': {[(int(i) + 1, label[i], round(float(sims[i]), 3)) for i in top]}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].barh(np.array(vocab)[stop][order][::-1], overall[order][::-1], color="grey", edgecolor="black")
ax[0].set_title("Most frequent stop words")
for c, st in zip(sorted(set(label)), ("o-", "s--", "^:")):
    rows = [i for i in range(N) if label[i] == c]
    mean = tfidf[rows].mean(axis=0)
    top = np.argsort(-mean)[:6]
    ax[1].plot(mean[top], st, color="black", label=c)
    content = np.where(~stop)[0]
    best = content[np.argsort(-mean[content])[:6]]
    print(f"top tf-idf words of {c:<12} with stop words: {', '.join(np.array(vocab)[top])}")
    print(f"{'':<27}{c:<12} without stop words: {', '.join(np.array(vocab)[best])}")
ax[1].set_xlabel("rank"); ax[1].set_ylabel("mean tf-idf"); ax[1].legend(fontsize=8); ax[1].set_title("Strongest words of every topic")
plt.tight_layout()
plt.show()
