import math
import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

GAMMA = 0.99


def network(n_out):
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_out))


def widen(state):
    return torch.tensor(np.concatenate([state, np.zeros(4 - len(state))]), dtype=torch.float32)


def energy(s):
    return 0.0025 * math.sin(3 * s[0]) + 0.5 * s[1] ** 2


class Agent:
    def __init__(self, env_id, net, trainable, seed, shaped):
        self.env, self.net, self.shaped = gym.make(env_id), net, shaped
        self.n = self.env.action_space.n
        self.target = network(self.n)
        self.target.load_state_dict(net.state_dict())
        self.opt = torch.optim.Adam(trainable, lr=1e-3)
        self.memory, self.seed = deque(maxlen=20000), seed
        random.seed(seed)
        torch.manual_seed(seed)

    def learn(self):
        S, A, R, S2, T = zip(*random.sample(self.memory, 64))
        S, S2 = torch.stack(S), torch.stack(S2)
        A, R, T = torch.tensor(A), torch.tensor(R, dtype=torch.float32), torch.tensor(T, dtype=torch.float32)
        q = self.net(S).gather(1, A.view(-1, 1)).squeeze()
        with torch.no_grad():
            y = R + GAMMA * self.target(S2).max(1).values * (1 - T)
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def episode(self, ep, total_eps, limit=200):
        eps = max(0.05, 1 - ep / (0.5 * total_eps))
        s, _ = self.env.reset(seed=self.seed * 1000 + ep)
        furthest, steps, done, success = -1.2, 0, False, False
        while not done and steps < limit:
            if random.random() < eps:
                a = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    a = int(self.net(widen(s)).argmax())
            s2, r, term, trunc, _ = self.env.step(a)
            done, success = term or trunc, term
            if self.shaped:
                r += 200 * (GAMMA * energy(s2) - energy(s))
            self.memory.append((widen(s), a, r, widen(s2), float(term)))
            furthest, s, steps = max(furthest, s2[0]), s2, steps + 1
            if len(self.memory) >= 64:
                self.learn()
        if ep % 5 == 0:
            self.target.load_state_dict(self.net.state_dict())
        return furthest, steps, success


source = network(2)
agent = Agent("CartPole-v1", source, source.parameters(), seed=5, shaped=False)
stats = [agent.episode(e, 100, limit=300) for e in range(100)]
print("pre-training on CartPole-v1: mean steps in the last 10 episodes =", round(np.mean([x[1] for x in stats[-10:]]), 1))

modes = {"scratch": "all layers trained from random start",
         "frozen": "hidden layers copied and frozen, only the new output layer learns",
         "fine-tuned": "hidden layers copied, every layer keeps learning"}
EPISODES, SEEDS = 80, (11, 12)
curves, success_at = {m: [] for m in modes}, {m: [] for m in modes}
for mode in modes:
    for seed in SEEDS:
        net = network(3)
        if mode != "scratch":
            net.load_state_dict({k: v for k, v in source.state_dict().items() if not k.startswith("4")}, strict=False)
        trainable = list(net[4].parameters()) if mode == "frozen" else list(net.parameters())
        ag = Agent("MountainCar-v0", net, trainable, seed, shaped=True)
        run = [ag.episode(e, EPISODES) for e in range(EPISODES)]
        curves[mode].append([x[0] for x in run])
        success_at[mode].append(next((i + 1 for i, x in enumerate(run) if x[2]), None))

print("\nMountainCar-v0: the car must reach position 0.5 (state padded from 2 to 4 numbers, 3 actions)")
for mode, text in modes.items():
    c = np.array(curves[mode])
    print(f"{mode:<11}| {text}")
    print(f"           first success at episode {success_at[mode]} | furthest position, episodes 1-10: {c[:, :10].mean():.3f}, last 10: {c[:, -10:].mean():.3f}")

plt.figure(figsize=(7.5, 3.8))
for (mode, c), st in zip(curves.items(), ("--", ":", "-")):
    plt.plot(np.convolve(np.mean(c, axis=0), np.ones(5) / 5, mode="valid"), st, color="black", lw=1.6, label=mode)
plt.axhline(0.5, color="grey", lw=0.8)
plt.xlabel("episode"); plt.ylabel("furthest position")
plt.title("Transfer from CartPole to MountainCar")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
