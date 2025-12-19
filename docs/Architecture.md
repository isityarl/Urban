# System Architecture

## 1. Project Structure Overview

```text
Urban/
├── analysis/              # analysis and visualization notebook
├── back/
│   ├── agents/            # RL agents (Multi-agent DQN and PPO)
│   ├── app/               # Backend application (API & services)
│   │   ├── services/      # SUMO runners, helpers
│   │   ├── utils/         # Shared utilities
│   │   ├── config.py      # Global configuration
│   │   └── main.py        # Application run point
│   ├── baseline/          # Baseline statistics and reference runs
│   ├── data/              # SUMO configuration files, map
│   ├── res/
│   │   ├── logs/          # Training logs of DQN and PPO
│   │   └── models/        # Trained models
│   ├── RL_env/            # Environment–SUMO communication layer
│   │   ├── env.py
│   │   └── parallel_env.py
│   └── train/             # Training and evaluation scripts
├── docs/                  # All docs
├── front/                 # Frontend application
├── tests/                 # Unit and integration tests
├── requirements.txt
└── .gitignore
```

## 2. Reason for This Architecture Choice
This structure allows independent development of reinforcement learning algorithms, environment interaction, training pipelines, and analysis tools.  
It improves maintainability, scalability, and enables easier experimentation with different algorithms and configurations without tightly coupling system components.


## 3. System Components
### Frontend (`front/`)

- Provides user interaction and visualization
- Controls experiments and displays results
- Communicates with the backend via API calls


### Backend (`back/`)

The backend contains the core system logic and is divided into modular components.

#### Agents (`back/agents/`)

- Implements reinforcement learning algorithms:
  - Multi Agent Deep Q-Network (MADQN)
  - Proximal Policy Optimization (PPO)
- Each agent encapsulates:
  - Neural network architectures
  - Training logic
  - Action selection policies


#### Environment Layer (`back/RL_env/`)

- Manages interaction between Python and the SUMO simulator
- Components:
  - `env.py`: Single-environment interface
  - `parallel_env.py`: Parallel multi-environment execution
- Abstracts SUMO and TraCI communication from RL agents


#### Training & Evaluation (`back/train/`)

- Controls experiment execution
- Responsibilities:
  - Training loops
  - Model evaluation
  - Model saving and loading
- Connects agents with environments and logging mechanisms


#### Application Layer (`back/app/`)

- Backend service and API layer
- Components:
  - `main.py`: Entry point, runs the backend server using Uvicorn
  - `config.py`: Centralized configuration
  - `services/`: SUMO execution, runtime helpers, log parsing
  - `utils/`: Shared utility functions


#### Data & Resources

- `back/data/`: SUMO network files, routes, and configuration
- `back/res/logs/`: Training and evaluation logs
- `back/res/models/`: Saved trained models
- `back/baseline/`: Baseline statistics for performance comparison


## 4. Data Flow

1. The frontend or training script initiates an experiment.
2. The backend loads configuration and initializes the selected RL agent.
3. The environment layer starts the SUMO simulation via TraCI.
4. For each simulation step:
   - The current state is retrieved from SUMO
   - The agent selects an action
   - The action is applied to the simulator
   - Reward and next state are collected
5. Training metrics and logs are stored.
6. Results are analyzed using scripts in the `analysis/` directory and visualized in the frontend.


## 5. Data Storage

- **Logs:** Episode-level and step-level metrics (e.g., rewards, duration, waiting time)
- **Models:** Serialized neural network weights
- **Baselines:** Reference metrics for comparison

All data is stored in a structured, file-based format to ensure reproducibility and support offline analysis.


## 6. Technology Decisions

- **Python:** Core implementation language
- **PyTorch:** Deep reinforcement learning framework
- **SUMO + TraCI:** Traffic simulation and control interface
- **FastAPI / Uvicorn:** Backend API server
- **Pandas & Matplotlib:** Data analysis and visualization


## 7. Future Extensions

- Multi-agent traffic signal control
- Distributed training across multiple machines
- Real-time training dashboard
- Support for additional reinforcement learning algorithms
- Experiment tracking using a dedicated database
