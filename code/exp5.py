import matplotlib.pyplot as plt
import numpy as np

ROWS, COLS = 5, 5
WALLS = {(1, 1), (2, 3), (3, 1)}
PIT, GOAL = (1, 3), (4, 4)
ACTIONS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
SIDE = {"U": "LR", "D": "LR", "L": "UD", "R": "UD"}
GAMMA, SLIP = 0.9, 0.2
cells = [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in WALLS]


def move(s, a):
    r, c = s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1]
    return s if not (0 <= r < ROWS and 0 <= c < COLS) or (r, c) in WALLS else (r, c)


def outcomes(s, a):
    if s in (PIT, GOAL):
        return []
    res = [(1 - SLIP, move(s, a))] + [(SLIP / 2, move(s, b)) for b in SIDE[a]]
    return [(p, s2, 10.0 if s2 == GOAL else -10.0 if s2 == PIT else -0.1) for p, s2 in res]


def evaluate(policy, theta=1e-6):
    V = {s: 0.0 for s in cells}
    sweeps = 0
    while True:
        delta = 0
        for s in cells:
            v = sum(p * (r + GAMMA * V[s2]) for p, s2, r in outcomes(s, policy[s]))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        sweeps += 1
        if delta < theta:
            return V, sweeps


def improve(V):
    return {s: max(ACTIONS, key=lambda a: sum(p * (r + GAMMA * V[s2]) for p, s2, r in outcomes(s, a))) if s not in (PIT, GOAL) else "-" for s in cells}


def show(grid, fmt):
    for r in range(ROWS):
        print("  " + " ".join(fmt(r, c) for c in range(COLS)))


policy = {s: ("R" if s[1] < 4 else "D") for s in cells}
V0, sw = evaluate(policy)
start_value = V0[(0, 0)]
print(f"grid {ROWS}x{COLS}, walls {sorted(WALLS)}, pit {PIT}, goal {GOAL}, discount {GAMMA}, slip {SLIP}")
print(f"1. evaluate the simple policy 'go right, then down': {sw} sweeps, value of the start square = {start_value:.2f}")
show(V0, lambda r, c: "  ####" if (r, c) in WALLS else f"{V0[(r, c)]:6.1f}")

history = [start_value]
for it in range(1, 12):
    new = improve(V0)
    V0, sw = evaluate(new)
    changed = sum(new[s] != policy[s] for s in cells)
    policy = new
    history.append(V0[(0, 0)])
    print(f"\n2. improvement round {it}: {changed} squares changed their action, evaluation {sw} sweeps, start value {V0[(0, 0)]:.2f}")
    if changed == 0:
        break

print("\nfinal policy (# wall, P pit, G goal)")
show(policy, lambda r, c: "#" if (r, c) in WALLS else "P" if (r, c) == PIT else "G" if (r, c) == GOAL else policy[(r, c)])
print("\nfinal state values")
show(V0, lambda r, c: "  ####" if (r, c) in WALLS else f"{V0[(r, c)]:6.1f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
grid = np.full((ROWS, COLS), np.nan)
for s in cells:
    grid[s] = V0[s]
ax[0].imshow(grid, cmap="Greys")
for s in cells:
    ax[0].text(s[1], s[0], f"{V0[s]:.1f}" if s in (PIT, GOAL) else f"{V0[s]:.1f}\n{policy[s]}", ha="center", va="center", fontsize=8, color="white" if V0[s] < np.nanmedian(grid) else "black")
ax[0].set_title("Final values and actions"); ax[0].set_xticks([]); ax[0].set_yticks([])
ax[1].plot(range(len(history)), history, "o-", color="black")
ax[1].set_xlabel("improvement round"); ax[1].set_ylabel("value of start square")
ax[1].set_title("Policy iteration improves the start value")
plt.tight_layout()
plt.show()
