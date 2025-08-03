# **Reinforcement Learning Project**

## Important Links:
[![YouTube Video tPNbfBhom8M](https://img.youtube.com/vi/tPNbfBhom8M/maxresdefault.jpg)](https://www.youtube.com/watch?v=tPNbfBhom8M)
1. Demo-Video: [Click here for Demo Video](https://youtu.be/tPNbfBhom8M)
2. Comprehensive Report: [Click here for Documentation Report](https://github.com/M-Pascal/Pascal_Mugisha_rl_Summative/blob/main/Report_ML_Techniques_II%20%5BPascal%20M_Summative_Assign%5D.pdf)

## Diabetes Treatment RL Environment

A comprehensive reinforcement learning project implementing and comparing multiple RL algorithms for optimal diabetes treatment decision-making in a simulated clinical environment.

## Project Overview

This project addresses the critical challenge of diabetes management through intelligent treatment decisions. The custom environment simulates a patient's daily life where an RL agent must navigate between different treatment options (insulin, medication, food choices) to maintain optimal blood glucose levels.

### Environment Details

- **Domain**: Medical/Healthcare - Diabetes Treatment Simulation
- **Agent Goal**: Maintain blood glucose levels within target range (80-120 mg/dL)
- **Actions**: 4 discrete actions (Up, Down, Left, Right) for navigating treatment options
- **Observations**: [agent_x, agent_y, sugar_level, time_hours] - 4D continuous space
- **Visualization**: Real-time pygame-based 2D environment with medical graphics

### Treatment Options

- **High-dose**:  glucose reduction (emergency situations)
- **Low-dose**:  glucose reduction (mild management)
- **Rapid glucose**: glucose increase (hypoglycemia treatment)
- **Moderate**:  glucose increase (natural energy)
- **Balanced**: Balanced nutrition (steady management)
- **Stop**: No treatment (maintenance periods)

## Implemented RL Algorithms

### Value-Based Method

- **DQN (Deep Q-Network)**: Classic value-based approach with experience replay and target networks

### Policy Gradient Methods

- **REINFORCE**: Basic policy gradient with Monte Carlo returns
- **PPO (Proximal Policy Optimization)**: Advanced policy gradient with clipped surrogate objective

### Actor-Critic Method

- **A2C (Advantage Actor-Critic)**: Combines value estimation with policy learning

## Project Structure

```
Pascal_Mugisha_rl_Summative/
├── environment/
│   ├── custom_env.py             # Custom Gymnasium environment
│   ├── rendering.py              # Pygame visualization system
│   ├── save_demo.py              # Random action demonstration
│   ├── results_random_action.png # Random action results visualization
│   ├── image/                    # Medical graphics assets
│   └── __pycache__/              # Python cache files
├── training/
│   ├── dqn_training.py           # DQN training implementation
│   ├── ppo_training.py           # PPO training implementation
│   ├── a2c_training.py           # A2C training implementation
│   ├── reinforce_training.py     # REINFORCE training implementation
│   ├── models/                   # Saved trained models
│   └── logs/                     # Training logs and metrics
├── analysis_plots/
│   ├── performance_comparison.png # Algorithm performance comparison
│   ├── training_curves.png       # Training progress visualization
│   ├── performance_summary.csv   # Detailed performance metrics
│   └── training_analysis.csv     # Training characteristics analysis
├── myenv/                        # Python virtual environment
├── main.py                       # Universal model runner with GIF recording
├── analysis.py                   # Comprehensive performance analysis
├── diabetes_simulation.gif       # Generated simulation recording
├── random_agent_demo.gif         # Random action demonstration GIF
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Quick Start for the project

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/M-Pascal/Pascal_Mugisha_rl_Summative.git
cd Pascal_Mugisha_rl_Summative

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Demo (Random Actions)

```bash
# Run random action demonstration (no trained model)
cd environment
python save_demo.py
```

This creates `random_agent_demo.gif` showing the environment visualization.

### 3. Train RL Models

```bash
cd training

# Train DQN
python dqn_training.py

# Train PPO
python ppo_training.py

# Train A2C
python a2c_training.py

# Train REINFORCE
python reinforce_training.py
```

### 4. Run Trained Models

```bash
# Run with specific model
python main.py --model training/models/dqn/dqn_diabetes_final --type dqn

# Run with PPO model
python main.py --model training/models/ppo/ppo_diabetes_final --type ppo

# List available models
python main.py --list-models

# Run with default DQN model (creates GIF automatically)
python main.py
```

### 5. Performance Analysis

```bash
# Generate comprehensive performance comparison
python analysis.py

# Or use simplified analysis with synthetic data
python simple_analysis.py
```

This creates analysis outputs in the `analysis_plots/` directory:

- `performance_comparison.png` - Algorithm performance comparison charts
- `training_curves.png` - Training progress visualization
- `performance_summary.csv` - Detailed performance metrics table
- `training_analysis.csv` - Training characteristics and insights

## Key Features

### GIF Recording

- Automatic GIF generation during simulation runs (`diabetes_simulation.gif`)
- Random action demonstration GIF (`random_agent_demo.gif`)
- Configurable duration and frame rate
- Perfect for assignment documentation and visual evidence

### Performance Metrics

- Episode rewards and lengths
- Blood glucose control effectiveness
- Time-in-target-range analysis (80-120 mg/dL)
- Algorithm comparison and ranking
- Training curve analysis with convergence metrics
- Sample efficiency and stability measurements

### Analysis Tools

- **Comprehensive Analysis** (`analysis.py`): Full model evaluation with real trained models
- **Simplified Analysis** (`simple_analysis.py`): Synthetic data comparison when models unavailable
- **Automated Visualization**: Performance graphs, training curves, and summary tables
- **CSV Export**: Detailed metrics for further analysis

### Clinical Relevance

- Realistic blood glucose dynamics
- Time-based treatment effects
- Emergency situation handling
- Multiple treatment modalities

### Hyperparameter Configurations

#### DQN Configuration

```python
learning_rate=3e-4
buffer_size=100000
batch_size=32
gamma=0.99
target_update_interval=10000
exploration_fraction=0.2
net_arch=[64, 64]
```

#### PPO Configuration

```python
learning_rate=3e-4
n_steps=2048
batch_size=64
gamma=0.99
gae_lambda=0.95
clip_range=0.2
net_arch=[64, 64]
```

## Results Summary

The project evaluates each algorithm on:

- **Mean Episode Reward**: Overall performance metric
- **Time in Target Range**: Clinical effectiveness (target: >70%)
- **Episode Length**: Efficiency of treatment decisions
- **Convergence Speed**: Training efficiency

## Assignment Compliance

- **Custom Environment**: Diabetes treatment simulation with medical relevance
- **Exhaustive Actions**: 4 discrete navigation actions fully explored
- **Advanced Visualization**: Pygame-based 2D environment with medical graphics
- **Random Action Demo**: `save_demo.py` shows environment without trained models
- **GIF Documentation**: Automatic GIF generation for visual evidence
- **Four RL Algorithms**: DQN, PPO, A2C, REINFORCE all implemented
- **Hyperparameter Analysis**: Detailed configuration and impact discussion
- **Performance Comparison**: Comprehensive analysis with visualizations
- **GitHub Repository**: Proper structure and documentation

## Technical Implementation

### Environment Specifications

- **Action Space**: `Discrete(4)` - Up/Down/Left/Right navigation
- **Observation Space**: `Box([0,0,40,0], [6,6,300,24])` - Position, glucose, time
- **Reward Function**: Multi-component reward encouraging glucose stability
- **Episode Termination**: Time limit or critical glucose levels

### Model Architecture

- **DQN**: MLP with [64, 64] hidden layers, experience replay, target networks
- **PPO**: Actor-critic with shared [64, 64] features, clipped surrogate loss
- **A2C**: Synchronous actor-critic with advantage estimation
- **REINFORCE**: Policy gradient with Monte Carlo returns

## Usage Examples

```bash
# Basic simulation with GIF recording
python main.py

# Generate analysis plots for report
python analysis.py

# Create random action demo
cd environment && python save_demo.py
```

## Usage Context

This project was developed for the RL Summative Assignment comparing value-based and policy gradient methods in a domain-specific healthcare application. The diabetes treatment environment provides a realistic testbed for evaluating RL algorithms' ability to make sequential decisions with clinical impact.


---

**Author**: Pascal Mugisha  
**Course**: ML_Technique_II  
**Environment**: Diabetes Treatment Simulation  
**Algorithms**: DQN, PPO, A2C, REINFORCE
