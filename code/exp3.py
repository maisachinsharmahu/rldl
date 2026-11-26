import re

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def plain(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def train(slippery, episodes=8000, seed=0):
    env = gym.make("FrozenLake-v1", is_slippery=slippery)
    rng = np.random.default_rng(seed)
    Q = np.zeros((16, 4))
    wins = []
    for ep in range(episodes):
        eps = max(0.05, 1 - ep / (episodes * 0.6))
        s, _ = env.reset(seed=int(rng.integers(1e9)))
        done = False
        while not done:
            a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s] + rng.random(4) * 1e-9))
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            Q[s, a] += 0.2 * (r + 0.95 * Q[s2].max() * (not term) - Q[s, a])
            s = s2
        wins.append(r)
    return env, Q, wins


def success_rate(env, Q, n=500):
    ok = 0
    for k in range(n):
        s, _ = env.reset(seed=20000 + k)
        for _ in range(100):
            s, r, term, trunc, _ = env.step(int(Q[s].argmax()))
            if term or trunc:
                ok += r
                break
    return ok / n


demo = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
demo.reset(seed=0)
print("FrozenLake-v1: a 4 x 4 frozen lake; S start, F frozen, H hole, G goal")
print(plain(demo.render()))
print("states: 16   actions: 4 (left, down, right, up)   reward: 1 only at the goal\n")

results = {}
for name, slippery in (("slippery ice", True), ("firm ice", False)):
    env, Q, wins = train(slippery)
    results[name] = (Q, wins)
    print(f"{name:<13}: success rate of the greedy policy over 500 games = {success_rate(env, Q) * 100:.1f} %"
          f"   (training wins in last 1000 episodes: {int(np.sum(wins[-1000:]))})")

Q = results["firm ice"][0]
arrows = "<v>^"
print("\nlearned policy on firm ice (H = hole, G = goal)")
holes, goal = {5, 7, 11, 12}, 15
for r in range(4):
    print("  " + " ".join("H" if 4 * r + c in holes else "G" if 4 * r + c == goal else arrows[int(Q[4 * r + c].argmax())] for c in range(4)))
print("\nbest value at the start square: firm ice", round(float(Q[0].max()), 3), "| slippery ice", round(float(results["slippery ice"][0][0].max()), 3))

plt.figure(figsize=(7.5, 3.8))
for name, st in (("firm ice", "-"), ("slippery ice", "--")):
    w = np.convolve(results[name][1], np.ones(200) / 200, mode="valid")
    plt.plot(w, st, color="black", label=name)
plt.xlabel("episode"); plt.ylabel("win rate (moving average)")
plt.title("Q-learning on FrozenLake"); plt.legend()
plt.tight_layout()
plt.show()
