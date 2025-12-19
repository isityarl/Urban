# UrbanIQ

## What is UrbanIQ?

**UrbanIQ** is a reinforcement learning platform for traffic signal control.  
It allows users to simulate urban traffic, evaluate different control strategies (MADQN, PPO, Fixed-Time baseline), and analyze performance metrics.  

---

## Target Users

- Traffic engineers and urban planners
- Researchers in reinforcement learning and intelligent transportation
- Developers experimenting with traffic simulation and optimization

---

## Tech Stack

- **Python 3.10+** – core language  
- **FastAPI / Uvicorn** – backend API server  
- **SUMO + TraCI** – traffic simulation  
- **PyTorch** – deep reinforcement learning  
- **Pandas & Matplotlib** – data analysis and visualization  
- **Frontend** – static files served via FastAPI  
- **others** - all needed libraries -> `requirements.txt`
---

## Project Structure
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

---

## Build and Installation

### System Requirements


- Python 3.10+
- SUMO installed and added to `PATH` (can be installed via [GitHub](https://github.com/eclipse-sumo/sumo?tab=readme-ov-file) or [official site](https://sumo.dlr.de/docs/Downloads.php))
- Required Python packages

### Setup Commands

```bash
git clone https://github.com/isityarl/Urban.git
cd Urban
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```