# TESP Snake Robot - Reinforcement Learning Navigation
Students: 
- Charlotte Primiceri
- Serena Trovalusci
- Davide De Ciutiis
- Harini Satyavada
- Vishnu Anand
  
An autonomous snake robot navigation system using Reinforcement Learning in MuJoCo simulation. Our team implemented an autonomous control through the Soft Actor-Critic (SAC) algorithm.

See our presentation IntelligentSnake.pdf for details!
## Overview

The 2024 team developed an interactive game using MediaPipe hand tracking to control a simulated snake robot. The 2025 iteration transitions from manual control to autonomous navigation using reinforcement learning in a maze environment.

## Features

- Autonomous snake robot navigation using SAC algorithm
- MuJoCo physics simulation
- Dynamic maze generation with A* pathfinding
- Central Pattern Generator (CPG) for biologically-inspired movement
- TensorBoard integration for training monitoring
- Graceful model saving and interruption handling

## Installation

### Create a Conda environment with Python 3.11:
```
conda create -n tesp2025 python=3.11
conda activate tesp2025
```


### Install dependencies:
``` 
 pip install -r requirements.txt
```


## Project Structure

project_root/  
├── assets/ # Assets (if applicable)  
├── scenes/ # MuJoCo XML simulation files   
├── models/ # Trained model weights (.zip files)  
├── stable_baselines3/ # Training scripts and configuration  
├── mazes/  
│ ├── make_maze.py # Maze generation and A* algorithm  
│ └── mujoco_tools.py # Dynamic XML generation for mazes  
├── cpg/  
│ └── snake_cpg.py # Central Pattern Generator for joint control  
├── env_snake.py # Custom Gym environment  
└── train_sac.py # SAC training script

## Usage

### Training

Start training the snake robot:
  python3 train_sac.py

This will:
- Initialize the MuJoCo environment
- Begin SAC training
- Log statistics to TensorBoard
- Save model weights to `models/` directory

(Video)

### Running Trained Mode
(Video)

### Monitoring Training

Launch TensorBoard to monitor training progress:

```
tensorboard --logdir sac_snake_tensorboard/
```

This will allow you to open http://localhost:6006 

## Technical Details

### Algorithm
- **Soft Actor-Critic (SAC)**: Chosen for stability and performance in high-dimensional continuous control tasks

### Environment
- Training occurs in a corridor environment rather than full maze due to computational complexity
- Custom reward function includes distance-based rewards, waypoint bonuses, and collision penalties

### Model Saving
- `sac_snake_final.zip`: Saved upon successful training completion
- `sac_snake_interrupted.zip`: Saved if training is interrupted

## System Requirements

- Python 3.11
- GPU support optional (recommended for faster training)
- Dependencies as listed in `requirements.txt`

## Limitations

- Training environment prone to computational instability in complex scenarios
- Currently limited to corridor environments (straight line movement) to maintain training stability
- Simulation for learning in a full maze produces NaN errors in complex maze configurations

## Future Work

- Training to enable full maze navigation (first with the aid of A* waypoints, then without any waypoints)
- Imporived rewards and penalty (however makes the learning more time/step consuming)
- Memory and vision modules integration
- Improved simulation stability

## References

- [MuJoCo Physics Engine](https://mujoco.org/)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [OpenAI Spinning Up - SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- [Hierarchical RL-Guided Large-scale Navigation of a Snake Robot](https://arxiv.org/abs/2312.03223v1#S5)










