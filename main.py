
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

#env creation
env = gym.make("FrozenLake-v1", map_name="8x8")

#set
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.99995
min_epsilon = 0.05
episodes = 100000

q_table = np.zeros(
    (env.observation_space.n, env.action_space.n)
)

rewards = []
for episode in range(episodes):

    state, info = env.reset()

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

        #Calc Q value
        best_next_q = np.max(q_table[next_state])
        #Update Q
        if terminated:
            target = reward
        else:
            target = reward + gamma * best_next_q

        q_table[state, action] += alpha * (
            target - q_table[state, action]
        )

        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    #progress print
    if (episode + 1) % 1000 == 0:
        recent_average = np.mean(rewards[-1000:])
        print(
            f"Episode {episode + 1}: "
            f"average reward = {recent_average:.3f}, "
            f"epsilon = {epsilon:.3f}"
        )

np.save("results/q_table.npy", q_table)
env.close()

plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.3, label="Episode reward")

window = 200
if len(rewards) >= window:
    moving_average = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )
    plt.plot(
        range(window - 1, len(rewards)),
        moving_average,
        label="200-episode moving average"
    )

plt.xlabel("Training Episode")
plt.ylabel("Reward")
plt.title("FrozenLake: Q-learning Baseline")
plt.legend()
plt.tight_layout()
plt.savefig("baseline_rewards.png")
plt.show()

print("\nTraining complete!")
print("Final Q-table shape:", q_table.shape)
print("Last 1000 episodes average reward:",
      np.mean(rewards[-1000:]))
print("Plot saved as baseline_rewards.png")


eval_env = gym.make("FrozenLake-v1", map_name="8x8")

evaluation_episodes = 100
evaluation_rewards = []
evaluation_steps = []

for episode in range(evaluation_episodes):
    state, info = eval_env.reset()

    total_reward = 0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated, info = (
            eval_env.step(action)
        )
        total_reward += reward
        steps += 1

    evaluation_rewards.append(total_reward)
    evaluation_steps.append(steps)

eval_env.close()

print("\nEvaluation results")
print("Episodes:", evaluation_episodes)
print("Mean reward:", np.mean(evaluation_rewards))
print("Success rate:", np.mean(evaluation_rewards))
print("Average steps:", np.mean(evaluation_steps))
