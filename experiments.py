
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

GAMMA = 0.99
INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.99995
MIN_EPSILON = 0.05
EPISODES = 100000

LEARNING_RATES = [0.1, 0.5, 0.9]

def train(alpha, seed=42):
    env = gym.make("FrozenLake-v1", map_name="8x8")

    np.random.seed(seed)
    env.action_space.seed(seed)

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
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            if terminated:
                target = reward
            else:
                target = reward + GAMMA * np.max(q_table[next_state])

            q_table[state, action] += alpha * (
                target - q_table[state, action]
            )

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

    env.close()
    return q_table, rewards

plt.figure(figsize=(10, 6))

for alpha in LEARNING_RATES:
    print(f"\nTraining with alpha = {alpha}")
    q_table, rewards = train(alpha)

    window = 1000
    averages = [
        np.mean(rewards[i:i + window])
        for i in range(0, len(rewards), window)
    ]

    x = np.arange(1, len(averages) + 1) * window
    plt.plot(x, averages, label=f"alpha = {alpha}")

    np.save(f"results/q_table_alpha_{alpha}.npy", q_table)

    print("Last 1000 episodes average reward:",
          np.mean(rewards[-1000:]))

plt.xlabel("Training Episode")
plt.ylabel("Average Reward (1000 episodes)")
plt.title("FrozenLake: Learning Rate Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("results/learning_rate_comparison.png")
plt.show()

print("\nSaved results/learning_rate_comparison.png")
