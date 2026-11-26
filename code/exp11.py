import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
from collections import deque

import gymnasium as gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

keras.utils.set_random_seed(11)
random.seed(11)
env = gym.make("CartPole-v1")
print("CartPole-v1 | state:", env.observation_space.shape[0], "numbers | actions:", env.action_space.n, "(push left, push right)")
print("an episode ends when the pole tilts more than 12 degrees, the cart leaves the track or 500 steps pass")


def build():
    return keras.Sequential([
        keras.layers.Input((4,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(2),
    ])


model, target = build(), build()
target.set_weights(model.get_weights())
optimizer = keras.optimizers.Adam(1e-3)
huber = keras.losses.Huber()
GAMMA, BATCH = 0.98, 64


@tf.function
def train_step(S, A, R, S2, T):
    best_next = tf.reduce_max(target(S2), axis=1)
    y = R + GAMMA * best_next * (1.0 - T)
    with tf.GradientTape() as tape:
        q = tf.reduce_sum(model(S) * tf.one_hot(A, 2), axis=1)
        loss = huber(y, q)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def greedy(state):
    return int(tf.argmax(model(state[None].astype("float32"), training=False)[0]))


replay = deque(maxlen=40000)
history, steps_done = [], 0
for ep in range(260):
    eps = max(0.03, 1.0 - ep / 120)
    s, _ = env.reset(seed=ep)
    total, done = 0.0, False
    while not done:
        a = env.action_space.sample() if random.random() < eps else greedy(s)
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        replay.append((s, a, r, s2, float(term)))
        s, total, steps_done = s2, total + r, steps_done + 1
        if len(replay) >= 500:
            S, A, R, S2, T = zip(*random.sample(replay, BATCH))
            train_step(tf.constant(np.array(S), tf.float32), tf.constant(A), tf.constant(R, tf.float32),
                       tf.constant(np.array(S2), tf.float32), tf.constant(T, tf.float32))
        if steps_done % 400 == 0:
            target.set_weights(model.get_weights())
    history.append(total)
    if (ep + 1) % 26 == 0:
        print(f"episodes {ep - 24:>3}-{ep + 1:<3} mean reward {np.mean(history[-26:]):6.1f}  best {max(history[-26:]):5.0f}  epsilon {eps:.2f}")

scores = []
for k in range(20):
    s, _ = env.reset(seed=7000 + k)
    total, done = 0.0, False
    while not done:
        s, r, term, trunc, _ = env.step(greedy(s))
        done = term or trunc
        total += r
    scores.append(total)
print(f"\nevaluation of the greedy policy on 20 new episodes: mean {np.mean(scores):.1f}, median {np.median(scores):.0f}, best {max(scores):.0f}")
print("network parameters:", model.count_params(), "| replay memory holds", len(replay), "steps | total steps trained:", steps_done)

plt.figure(figsize=(7.5, 3.8))
plt.plot(history, color="lightgrey")
plt.plot(np.convolve(history, np.ones(15) / 15, mode="valid"), color="black")
plt.xlabel("episode"); plt.ylabel("reward")
plt.title("DQN on CartPole with Keras and TensorFlow")
plt.tight_layout()
plt.show()
