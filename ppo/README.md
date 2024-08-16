# Enhancing Analogical Reasoning in the Abstraction and Reasoning Corpus via Model-Based RL
The implementation of experiments comparing Proximal Policy Optimization (PPO) and DreamerV3 within the ARCLE environment.


## Algorithm Details

### Proximal Policy Optimization (PPO)
- [paper](https://arxiv.org/pdf/1707.06347)
- [code](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py)

We reimplemented the experiments from the first implementation of PPO on ARCLE environments.
- [code](https://github.com/ku-dmlab/arc_trajectory_generator)

### Mastering Diverse Domains through World Models (Dreamerv3)
[paper](https://arxiv.org/pdf/2301.04104v1)
[code](https://github.com/NM512/dreamerv3-torch)

This is pythorch implementation of authors' DreamerV3 implementation 
- [code]


## Experimental setting
- Actions - We used 5 operation, and entire selection
- Tasks - We selected 4 tasks that can be solved with entire selection
- 사진`

## Research Question

 RQ1: Learning a Single Task
 RQ2: Reasoning about Tasks Similar to Pre-Trained Task
 RQ3: Reasoning about Sub-Tasks of Pre-Trained Task
 RQ4: Learning Multiple Tasks Simultaneously 
 RQ5: Reasoning about Merged-Tasks of Pre-Trained Tasks

In this work, we test RQ1 and RQ2

## Results

# Instructions
The code has been tested on Linux and Mac and requires Python 3.11+.

# Acknowledgement

# Disclaimer
