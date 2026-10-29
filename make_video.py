
import gymnasium as gym
import numpy as np
import imageio
import os

q_table = np.load("results/q_table_decay_0.9999.npy")

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="rgb_array")
frames = []

state, _ = env.reset(seed=42)
done = False
truncated = False
frames.append(env.render())

while not (done or truncated):
    action = int(np.argmax(q_table[state]))
    state, reward, done, truncated, _ = env.step(action)
    frames.append(env.render())

env.close()

os.makedirs("results", exist_ok=True)
imageio.mimsave("results/frozenlake_agent.mp4", frames, fps=3)
print("Video saved to results/frozenlake_agent.mp4")
