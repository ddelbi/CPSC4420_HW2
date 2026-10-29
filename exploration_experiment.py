
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

#settings
ALPHA = 0.1
GAMMA = 0.99
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPISODES = 100000

DECAY_RATES = [0.9999, 0.99995, 0.99999]


def train(epsilon_decay, seed=42):
    env = gym.make("FrozenLake-v1", map_name="8x8")
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)

    q_table = np.zeros(
        (env.observation_space.n, env.action_space.n)
    )

    epsilon = INITIAL_EPSILON
    rewards = []

    for episode in range(EPISODES):
        state, info = env.reset(seed=seed + episode)
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            if terminated:
                target = reward
            else:
                target = reward + GAMMA * np.max(
                    q_table[next_state]
                )

            q_table[state, action] += ALPHA * (
                target - q_table[state, action]
            )

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(
            MIN_EPSILON, epsilon * epsilon_decay
        )

    env.close()
    return q_table, rewards


plt.figure(figsize=(10, 6))

for decay in DECAY_RATES:
    print(f"\nTraining with epsilon decay = {decay}")

    q_table, rewards = train(decay)

    window = 1000
    averages = [
        np.mean(rewards[i:i + window])
        for i in range(0, len(rewards), window)
    ]

    x = np.arange(1, len(averages) + 1) * window
    plt.plot(x, averages, label=f"decay = {decay}")

    np.save(
        f"results/q_table_decay_{decay}.npy",
        q_table
    )

    print(
        "Last 1000 episodes average reward:",
        np.mean(rewards[-1000:])
    )

plt.xlabel("Training Episode")
plt.ylabel("Average Reward (1000 episodes)")
plt.title("FrozenLake: Epsilon Decay Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("results/epsilon_decay_comparison.png")
plt.show()

print("\nSaved results/epsilon_decay_comparison.png")
