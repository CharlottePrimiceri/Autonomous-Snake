# Intelligent Snake — Autonomous Robot Navigation via RL & CPG

An autonomous snake robot navigation system using Reinforcement Learning in MuJoCo simulation. Building on a gesture-controlled base from the 2024 edition, the 2025 iteration empowers the robot to navigate fully autonomously using a hierarchical control scheme combining A* global planning, SAC-based local planning, and a Central Pattern Generator for biologically-inspired locomotion.

> **Authors:** Charlotte L. Primiceri · Serena Trovalusci · Davide de Ciutiis · Harini Satyavada · Vishnu Anand  
> **Program:** TESP 2025 — Tohoku University  
> **Slides:** [`IntelligentSnake.pdf`](IntelligentSnake.pdf)

---

## What is this project?

Snake robots are ideal for exploring hard-to-reach or hazardous environments, from disaster zones to planetary surfaces. When operating on unknown complex terrains, real-time gait adjustment based on environmental perception is key for navigation.

This project takes a gesture-controlled snake robot and makes it fully autonomous, using artificial intelligence to **perceive, plan, and move without human intervention**:

```
┌─────────────────────────────────────────────────────────────────┐
│               Hierarchical Control Scheme                       │
│                                                                 │
│  Occupancy Grid                                                 │
│       │                                                         │
│       ▼                                                         │
│  Global Planning  ──  A* Algorithm  ──►  Waypoints              │
│                                              │                  │
│                                              ▼                  │
│  Local Planning   ──  SAC (RL)      ──►  CPG params (R,θ,ω,δ)  │
│                                              │                  │
│                                              ▼                  │
│  Gait Generation  ──  CPG           ──►  Joint targets x[i]     │
│                                              │                  │
│                                              ▼                  │
│  Gait Tracking    ──  MuJoCo        ──►  Snake locomotion       │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Overview

| Component | Technology | Role |
|-----------|-----------|------|
| **Gesture Control** *(base)* | MediaPipe + TCP Socket | Human wrist tracking → snake direction |
| **Global Planning** | A* Algorithm | Computes waypoints from occupancy grid |
| **Local Planning** | SAC (Stable-Baselines3) | Learns CPG parameters to navigate toward waypoints |
| **Gait Generation** | Central Pattern Generator (CPG) | Translates parameters into sinusoidal joint trajectories |
| **Simulation** | MuJoCo | Physics engine for training and evaluation |

---

## Repository Structure

```
.
├── env_snake.py              # Custom Gym environment
├── train_sac.py              # SAC training script
├── requirements.txt
├── scenes/                   # MuJoCo XML simulation files
├── models/                   # Trained model weights (.zip files)
│   ├── sac_snake_final.zip
│   └── sac_snake_interrupted.zip
├── stable_baselines3/        # Training scripts and configuration
├── mazes/
│   ├── make_maze.py          # Maze generation and A* algorithm
│   └── mujoco_tools.py       # Dynamic XML generation for mazes
└── cpg/
    └── snake_cpg.py          # Central Pattern Generator for joint control
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create environment and install dependencies

```bash
conda create -n tesp2025 python=3.11
conda activate tesp2025
pip install -r requirements.txt
```

### 3. Train the snake

```bash
python3 train_sac.py
```

This will initialize the MuJoCo environment, begin SAC training, log statistics to TensorBoard, and save model weights to `models/`.

### 4. Monitor training

```bash
tensorboard --logdir sac_snake_tensorboard/
```

Then open [http://localhost:6006](http://localhost:6006).

---

## Technical Details

### Gesture-Controlled Snake *(base system)*

The 2024 base uses **MediaPipe** Pose Landmark Detection to track the right wrist position from a webcam feed. The wrist position `[x, y, is_hand_in_frame]` is streamed via a **TCP socket server** to the MuJoCo control system, which maps it to sinusoidal CPG parameters:

```
target q[i] = amp * sin(θ + φ * i) + B

ω = f(y)    angular velocity
B = f(x)    offset
φ = π/4     phase
θ_new = θ_old + ω·t
```

### SAC Algorithm *(local planning)*

The **Soft Actor-Critic** agent learns to navigate by interacting with a custom snake environment:

- **Observation space**: `2 × n_joints + head position`
- **Action space**: CPG parameters `R, θ, ω, δ` (full) or just `θ` (reduced, best results)
- **Reward**: reaching waypoints, getting closer to waypoint, turning head toward waypoint
- **Penalties**: jerky movements, self-collision, turning head away from waypoint

### Central Pattern Generator *(gait generation)*

The CPG is a biologically-inspired neural circuit that generates coordinated rhythmic joint trajectories. Joint targets follow a serpenoid wave: `x = r · sin(φ) + δ`, with CPG dynamics governed by a gradient system that ensures smooth, stable oscillations.

---

## Experiments

| Experiment | Environment | Action Space | Result |
|-----------|------------|-------------|--------|
| **Exp 1** | Maze | Full CPG: R, θ, ω, δ | Poor — unstable and inefficient behavior |
| **Exp 2** | Corridor (simplified) | Full CPG: R, θ, ω, δ | Still erratic movements |
| **Exp 3** *(best)* | Corridor | Reduced: θ only (R=1, ω=1, δ=0) | Improved stability and behavior |

Reducing the action space to a single learned parameter (θ, phase shift) while fixing the others significantly improved learning stability. The corridor environment was used instead of the full maze due to computational complexity.

---

## Demo

**Gesture-Controlled Snake** *(base system)*

https://github.com/user-attachments/assets/1f2fb36e-e6ef-432a-9bc7-e9d4c0159e0e

**Autonomous RL Snake** *(Experiment 3 — best performance)*

https://github.com/user-attachments/assets/5ae85dd3-5b0f-47d9-a050-bf8340631566

---

## Limitations

- Training environment prone to computational instability in complex scenarios.
- Currently limited to corridor environments — full maze produces NaN errors in complex configurations.
- Ring and pinky finger movements of the operator are not tracked in the gesture-controlled version.

---

## Future Work

- Training to enable full maze navigation (first with A* waypoints, then without any waypoints).
- Improved rewards and penalty functions.
- Memory and vision modules integration.
- Real-life bot to validate simulations.
- Improved simulation stability.

---

## References

- Jiang et al., [*Hierarchical RL-Guided Large-scale Navigation of a Snake Robot*](https://arxiv.org/abs/2312.03223v1), 2023
- Bing et al., *Smooth Gait Transition of Body Shape and Locomotion Speed Based on CPG Control For Snake-like Robot*, 2017
- Qiao et al., *Sigmoid transition approach of the central pattern generator-based controller for the snake-like robot*, 2016
- [MuJoCo Physics Engine](https://mujoco.org/)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
