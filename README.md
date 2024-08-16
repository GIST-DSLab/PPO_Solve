# Using PPO to solve ARC Problem

## Environments
```bash
conda create --name your_env_name python=3.9
```

To install pacakges
```bash
pip install -r requirements
```

Change the path in code.

change wandb id to yours, or delete.
in run.py line OOO


change the path to your directory.
in ppo.py line OOO
in loader.py line 111

change the task number
in loader.py line 112, 113

150 - 3x3 Horizontal flip task
241 - 3x3 diagonal flip task
179 - nxn diagonal flip task
380 - 3x3 CCW rotate task
