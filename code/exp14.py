import math
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

lines = [l.rstrip("\n").split("\t", 1) for l in open("data/SMSSpamCollection", encoding="utf-8", errors="ignore")]
labels = np.array([1 if a == "spam" else 0 for a, _ in lines])
texts = [b for _, b in lines]
print("messages:", len(texts), "| spam:", int(labels.sum()), "| ham:", int((1 - labels).sum()))


def clean(t):
    t = t.lower()
    t = re.sub(r"(https?://\S+|www\.\S+)", " urltoken ", t)
    t = re.sub(r"[£$]\s?\d+[\d,.]*", " moneytoken ", t)
    t = re.sub(r"\d{5,}", " longnumber ", t)
    t = re.sub(r"\d+", " number ", t)
    return re.findall(r"[a-z']+", t)


rng = np.random.default_rng(14)
order = rng.permutation(len(texts))
cut = int(0.8 * len(texts))
train, test = order[:cut], order[cut:]

word_counts = [Counter(), Counter()]
docs = [0, 0]
for i in train:
    word_counts[labels[i]].update(set(clean(texts[i])))
    docs[labels[i]] += 1
vocab = set(word_counts[0]) | set(word_counts[1])
V = len(vocab)
total = [sum(word_counts[c].values()) for c in (0, 1)]
prior = [math.log(docs[c] / sum(docs)) for c in (0, 1)]
print(f"training messages: {len(train)} (spam {docs[1]}, ham {docs[0]}) | vocabulary: {V} words")


def log_likelihood(word, c, alpha=1.0):
    return math.log((word_counts[c][word] + alpha) / (total[c] + alpha * V))


def spam_probability(text):
    ws = [w for w in set(clean(text)) if w in vocab]
    s = [prior[c] + sum(log_likelihood(w, c) for w in ws) for c in (0, 1)]
    m = max(s)
    e = [math.exp(x - m) for x in s]
    return e[1] / sum(e)


probs = np.array([spam_probability(texts[i]) for i in test])
pred = (probs > 0.5).astype(int)
y = labels[test]
tp, fp, fn, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
precision, recall = tp / (tp + fp), tp / (tp + fn)
print(f"\ntest messages: {len(test)}")
print(f"true spam caught {tp}, spam missed {fn}, ham marked as spam {fp}, ham passed {tn}")
print(f"accuracy {100 * (tp + tn) / len(test):.2f} % | precision {precision:.3f} | recall {recall:.3f} | F1 {2 * precision * recall / (precision + recall):.3f}")
print(f"a filter that calls everything ham would be {100 * (1 - y.mean()):.1f} % accurate, so accuracy alone is not enough")

ratio = {w: log_likelihood(w, 1) - log_likelihood(w, 0) for w in vocab if word_counts[0][w] + word_counts[1][w] >= 8}
ranked = sorted(ratio, key=ratio.get)
print("\nstrongest spam words:", ", ".join(ranked[::-1][:10]))
print("strongest ham words :", ", ".join(ranked[:10]))

print("\nthreshold study (how many ham messages are wrongly blocked for a given spam recall)")
for th in (0.5, 0.9, 0.99, 0.999):
    p = (probs > th).astype(int)
    print(f"  threshold {th:<6} recall {((p == 1) & (y == 1)).sum() / (y == 1).sum():.3f}  ham blocked {int(((p == 1) & (y == 0)).sum())}")

samples = ["Winner! You have been selected for a 1000 pound prize, call 09061701461 to claim",
           "Hey, are you coming to the lab tomorrow? bring the file",
           "FREE entry to our weekly competition, text WIN to 80086 now",
           "I will call you after dinner, save some food for me"]
print("\nnew messages")
for t in samples:
    p = spam_probability(t)
    print(f"  {'SPAM' if p > 0.5 else 'ham ':<4} {p:.3f}  {t[:64]}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].imshow([[tn, fp], [fn, tp]], cmap="Greys")
for (i, j), v in np.ndenumerate([[tn, fp], [fn, tp]]):
    ax[0].text(j, i, v, ha="center", va="center", color="white" if v > 500 else "black", fontsize=12)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["ham", "spam"]); ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["ham", "spam"])
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true"); ax[0].set_title("Naive Bayes written from scratch")
top = ranked[::-1][:10]
ax[1].barh(top[::-1], [ratio[w] for w in top[::-1]], color="grey", edgecolor="black"); ax[1].set_title("Strongest spam words")
ax[2].hist([probs[y == 0], probs[y == 1]], bins=20, color=["lightgrey", "black"], label=["ham", "spam"])
ax[2].set_yscale("log"); ax[2].set_title("Spam probability given to test messages"); ax[2].legend()
plt.tight_layout()
plt.show()
