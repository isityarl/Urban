import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from alpyne.constants import EngineState
from alpyne.sim import AnyLogicSim
from alpyne.utils import next_num
from DQN import DQN
from ReplayBuffer import ReplayBuffer

# ---------------- CONFIG ----------------
MODEL_PATH = 'rl_test1/Almaty.zip'
JAVA_PATH = '/usr/lib/jvm/java-21-openjdk/bin/java'
SIM_STOP_TIME = 3600  # seconds
NUM_EPISODES = 50
ACTION_DIM = 30
STATE_KEYS = [f'T{i}_{d}' for i in range(30) for d in ['N','S','E','W','Phase']]  # adjust based on your model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- HELPERS ----------------
def map_action_to_dict(action_array: np.ndarray):
    """Convert integer array to traffic light action dict."""
    return {f'Phase{i}': int(action_array[i]) for i in range(ACTION_DIM)}

def observation_to_state(obs_dict):
    """Flatten observation dict into a state vector in fixed order."""
    state = []
    for key in STATE_KEYS:
        state.append(obs_dict.get(key, 0.0))
    return np.array(state, dtype=np.float32)

# ---------------- DQN AGENT ----------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=50000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def select_action(self, state):
        if random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = F.mse_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ---------------- MAIN TRAINING LOOP ----------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    sim = AnyLogicSim(
        MODEL_PATH,
        java_path=JAVA_PATH,
        auto_finish=False,
        timeout_initial_connection=60000,
        max_server_await_time=60000,
        sim_params={'engine_overrides': {'stop_time': SIM_STOP_TIME, 'seed': next_num()}},
        config_defaults={}
    )

    # Initialize agent
    state_dim = len(STATE_KEYS)
    agent = DQNAgent(state_dim=state_dim, action_dim=ACTION_DIM)

    for episode in range(NUM_EPISODES):
        print(f"\n=== Episode {episode + 1} ===")
        status = sim.reset()

        # convert obs dict to state
        state = observation_to_state(status.observation)

        total_reward = 0
        step = 0

        while status.state != EngineState.FINISHED:
            action_idx = agent.select_action(state)
            action_array = np.zeros(ACTION_DIM, dtype=int)
            action_array[action_idx] = 1
            action_payload = map_action_to_dict(action_array)

            status = sim.take_action(**action_payload)
            next_state = observation_to_state(status.observation)

            # define reward (simple negative congestion sum example)
            densities = [v for k, v in status.observation.items() if k.endswith('_Q')]
            reward = -sum(densities)

            done = status.state == EngineState.FINISHED
            agent.memory.push(state, action_idx, reward, next_state, done)

            state = next_state
            total_reward += reward
            step += 1

            agent.update(batch_size=64)

        agent.sync_target()
        print(f"Episode {episode + 1} finished | Steps: {step} | Total Reward: {total_reward:.2f}")

    print("\nTraining finished. Closing simulation.")
    del sim
