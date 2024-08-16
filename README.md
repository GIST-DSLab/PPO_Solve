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


<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <div>
    <img width="173" alt="150" src="https://github.com/user-attachments/assets/5f68b706-51af-4416-977e-51044cf36ada">
  </div>
  <div>
    <img width="171" alt="380" src="https://github.com/user-attachments/assets/a55c2f2b-22f4-41c5-8942-8acd531f5685">
  </div>
  <div>
    <img width="170" alt="179" src="https://github.com/user-attachments/assets/ea04a9bd-4175-4ca5-9c51-f19682491e40">
  </div>
  <div>
    <img width="171" alt="380" src="https://github.com/user-attachments/assets/a55c2f2b-22f4-41c5-8942-8acd531f5685">
  </div>
</div>


# Acknowledge

This code is reimplemented based on https://github.com/ku-dmlab/arc_trajectory_generator