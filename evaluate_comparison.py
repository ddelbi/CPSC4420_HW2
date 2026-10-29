
import gymnasium as gym
import numpy as np

EPISODES = 100

def evaluate(q_table, seed=1000):
    env = gym.make("FrozenLake-v1", map_name="8x8")
    rewards = []
    steps_list = []

    for episode in range(EPISODES):
        state, info = env.reset(seed=seed + episode)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, info = (
                env.step(action)
            )
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)

    env.close()

    return np.mean(rewards), np.mean(steps_list)


# Load the baseline and best epsilon-decay Q-tables
baseline = np.load("results/q_table.npy")
best = np.load("results/q_table_decay_0.9999.npy")

baseline_reward, baseline_steps = evaluate(baseline)
best_reward, best_steps = evaluate(best)

print("\nEvaluation Results (100 episodes)")
print(f"Baseline success rate: {baseline_reward:.3f}")
print(f"Baseline average steps: {baseline_steps:.2f}")
print(f"Best agent success rate: {best_reward:.3f}")
print(f"Best agent average steps: {best_steps:.2f}")
print(f"Success-rate improvement: "
      f"{best_reward - baseline_reward:+.3f}")
