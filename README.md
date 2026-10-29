# FrozenLake Q-Learning — CPSC 4420 HW2

## Project Description

This project uses Q-learning to train an agent to navigate the 8×8 FrozenLake environment in Gymnasium. The agent learns to reach the goal while avoiding holes. Different learning rates, discount factors, and exploration decay rates are tested to compare performance.

## Files

- `main.py` — Trains and evaluates the baseline agent.
- `experiments.py` — Compares different learning rates.
- `discount_experiment.py` — Compares different discount factors.
- `exploration_experiment.py` — Compares different exploration decay rates.
- `evaluate_comparison.py` — Compares the baseline agent with the selected agent.
- `make_video.py` — Generates a video of the trained agent.
- `results/` — Contains experiment plots, saved Q-tables, and the agent video.

## Setup

Requires Python 3. Install the dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "gymnasium[toy-text]" numpy matplotlib imageio imageio-ffmpeg
```

## Running the Code

Run each script from the project directory:

```bash
python main.py
python experiments.py
python discount_experiment.py
python exploration_experiment.py
python evaluate_comparison.py
python make_video.py
```

The scripts generate experiment results and plots in the `results/` directory. The video is saved as `results/frozenlake_agent.mp4`.
