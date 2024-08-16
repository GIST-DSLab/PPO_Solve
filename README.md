# Using PPO to solve ARC Problem
Train ARC Task (number: 150, 179, 241, 380) with PPO agent.

# Instructions

## Environments
```bash
conda create --name your_env_name python=3.9
```

To install pacakges
```bash
pip install -r requirements.txt
```

## How to use

To run the example task (train task 150, eval 150)
```bash
python3 run.py train.task=150 eval.task=150
```

Choose the task within 150, 179, 241, 380

150 - 3 x 3 Horizontal flip task

179 - N x N diagonal flip task

241 - 3 x 3 diagonal flip task

380 - 3 x 3 CCW rotate task

![image](https://github.com/user-attachments/assets/138611b3-824f-47e2-a5ab-35f4362bb960)


# Acknowledge

This code is reimplemented based on https://github.com/ku-dmlab/arc_trajectory_generator