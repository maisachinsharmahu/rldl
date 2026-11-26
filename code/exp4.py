import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

random.seed(3)
np.random.seed(3)
torch.manual_seed(3)
env = gym.make("Acrobot-v1")
print("Acrobot-v1: two linked arms; swing the free end above the line")
print("state size:", env.observation_space.shape[0], " actions:", env.action_space.n, " reward: -1 per step until the goal")


def net():
    return nn.Sequential(nn.Linear(6, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, 3))


online, target = net(), net()
target.load_state_dict(online.state_dict())
opt = torch.optim.Adam(online.parameters(), lr=7e-4)
memory = deque(maxlen=30000)
gamma, batch, episodes = 0.99, 64, 110
scores = []

for ep in range(episodes):
    eps = max(0.03, 1 - ep / 60)
    s, _ = env.reset(seed=ep)
    total, done, steps = 0, False, 0
    while not done:
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(online(torch.tensor(s, dtype=torch.float32)).argmax())
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        memory.append((s, a, r, s2, float(term)))
        s, total, steps = s2, total + r, steps + 1
        if len(memory) >= 500 and steps % 2 == 0:
            S, A, R, S2, T = [torch.tensor(np.array(x), dtype=torch.float32) for x in zip(*random.sample(memory, batch))]
            q = online(S).gather(1, A.long().view(-1, 1)).squeeze()
            with torch.no_grad():
                best = online(S2).argmax(1, keepdim=True)
                y = R + gamma * target(S2).gather(1, best).squeeze() * (1 - T)
            loss = nn.functional.smooth_l1_loss(q, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if steps % 250 == 0:
            target.load_state_dict(online.state_dict())
    scores.append(total)
    if (ep + 1) % 10 == 0:
        print(f"episode {ep + 1:>3}  average reward of last 10 = {np.mean(scores[-10:]):7.1f}  epsilon = {eps:.2f}")

test = []
for k in range(20):
    s, _ = env.reset(seed=900 + k)
    total, done = 0, False
    while not done:
        with torch.no_grad():
            s, r, term, trunc, _ = env.step(int(online(torch.tensor(s, dtype=torch.float32)).argmax()))
        done = term or trunc
        total += r
    test.append(total)
print(f"\ngreedy policy over 20 test episodes: mean reward {np.mean(test):.1f}, best {max(test):.0f}, worst {min(test):.0f}")
print(f"first 10 training episodes averaged {np.mean(scores[:10]):.1f}; a random policy needs about 500 steps")

plt.figure(figsize=(7.5, 3.8))
plt.plot(scores, color="lightgrey")
plt.plot(np.convolve(scores, np.ones(8) / 8, mode="valid"), color="black")
plt.xlabel("episode"); plt.ylabel("reward (higher is better)")
plt.title("Double DQN on Acrobot")
plt.tight_layout()
plt.show()
