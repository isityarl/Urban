# Product Requirements Document

## 1. Product Goal

The goal of the product is to provide a *reinforcement learning–based traffic signal control system* that improves urban traffic flow by reducing *waiting time*, *congestion*, and *travel delays* using a real traffic simulation environment which is **Almaty's subnetwork**.

The system enables researchers and engineers to train, evaluate, and compare reinforcement learning algorithms in a city-scale traffic scenario simulated using SUMO.

---

## 2. Problem Statement

Urban traffic congestion leads to increased travel time, fuel consumption, and environmental impact. Traditional fixed-time or rule-based traffic signal control systems are unable to adapt to dynamic traffic conditions.

There is a need for an intelligent, adaptive traffic signal control solution that:
- Responds to real-time traffic conditions
- Learns optimal signal policies automatically
- Can be evaluated safely before real-world deployment

---

## 3. Target Audience

- Researchers working on traffic optimization
- Students studying reinforcement learning or intelligent transportation systems
- Engineers evaluating adaptive traffic control strategies
- Academic institutions and simulation-based research projects

---

## 4. Main User Roles

### 4.1 Researcher / Engineer
- Configures traffic scenarios
- Trains reinforcement learning models
- Evaluates and compares algorithm performance

### 4.2 System Operator
- Runs simulations
- Monitors training progress and metrics
- Manages trained models and logs

---

## 5. Core User Scenarios

1. **Training a Model**
   - User selects an RL algorithm (MADQN or PPO)
   - User configures training parameters
   - System runs SUMO simulations and trains the agent
   - Training metrics are logged and stored

2. **Evaluating a Trained Model**
   - User loads a trained model
   - System runs evaluation episodes
   - Performance metrics are collected and displayed

3. **Analyzing Results**
   - User accesses logs and statistics
   - User visualizes rewards, waiting time, and traffic metrics
   - User compares baseline and trained models

---

## 6. Functional Requirements

### FR-1: Traffic Simulation Execution
- The system shall start and control SUMO simulations via TraCI.
- The system shall support single and parallel simulation environments.

### FR-2: Reinforcement Learning Agents
- The system shall provide MADQN and PPO implementations.
- The system shall support training and inference modes.

### FR-3: Training Pipeline
- The system shall execute training episodes automatically.
- The system shall store episode-level metrics and rewards.

### FR-4: Model Management
- The system shall save trained models to disk.
- The system shall load saved models for evaluation.

### FR-5: Logging and Metrics
- The system shall log performance metrics such as:
  - Reward
  - Waiting time
  - Speed
  - Travel duration
- Logs shall be stored in a structured format.

### FR-6: Visualization and Analysis
- The system shall support offline analysis of results.
- The system shall generate plots of rewards and metrics over episodes.

---

## 7. Non-Functional Requirements

### Performance
- The system shall handle simulations with multiple intersections.
- Training shall support parallel environments to improve throughput.

### Reliability
- The system shall recover gracefully from simulation errors.
- Logs and models shall not be corrupted on failure.

### Usability
- Configuration files shall be easy to modify.
- Results shall be interpretable via plots and statistics.

### Scalability
- The system shall support increasing the number of intersections.
- The architecture shall allow adding new RL algorithms.

### Security
- The system shall not expose sensitive system-level resources.
- Access to model and log files shall be restricted to local execution.

---

## 8. MVP Scope (Version 1)

The MVP shall include the following features:

- SUMO integration via TraCI
- MADQN and PPO agent implementations
- Single- and multi-environment training
- Episode-level logging of rewards and traffic metrics
- Model saving and loading
- Offline analysis and visualization scripts
- Provide trip stats on fixed, MADQN, PPO models

---

## 9. Out-of-Scope Features

The following features are explicitly excluded from version 1:

- Real-world traffic signal deployment
- Online learning in live traffic systems
- Graphical real-time traffic visualization
- Multi-agent coordination between intersections
- Cloud-based or distributed training infrastructure

---

## 10. Acceptance Criteria

### AC-1: Training Execution
- Given a configuration file, the system successfully runs training episodes.
- Training logs are generated for each episode.

### AC-2: Model Persistence
- A trained model is saved after training completes.
- The saved model can be loaded and used for evaluation.

### AC-3: Simulation Control
- SUMO starts and stops correctly for each episode.
- No simulation processes remain after execution.

### AC-4: Metric Collection
- Reward and traffic metrics are recorded for every episode.
- Metrics can be parsed into a structured dataset.

### AC-5: Result Visualization
- Reward plots with moving averages can be generated.
- Performance metrics are visualized over episodes.

---

## 11. Success Metrics

- Reduction in average waiting time compared to baseline
- Improved average speed and reduced time loss
- Stable learning curves for MADQN and PPO agents
- Successful execution of multiple training runs without failure
