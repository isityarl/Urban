import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from back.agents.DQN import DQN

class BaseAgent:
    def __init__(self, state_size, tls_phases, config, rl_action_counts=None):
        self.state_size = state_size
        self.tls_phases = tls_phases
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rl_action_counts = rl_action_counts or self._derive_rl_action_counts(tls_phases)

        self.policy_net = {}
        self.target_net = {}
        self.optimizer = {}
        self.memory = {}

        for tl, phases in tls_phases.items():
            action_size = self.rl_action_counts.get(tl, max(1, len(phases)))
            self.policy_net[tl] = DQN(state_size, action_size).to(self.device)
            self.target_net[tl] = DQN(state_size, action_size).to(self.device)
            self.target_net[tl].load_state_dict(self.policy_net[tl].state_dict())
            self.target_net[tl].eval()
            self.optimizer[tl] = optim.AdamW(self.policy_net[tl].parameters(), lr=config['learning_rate'])
            self.memory[tl] = deque(maxlen=config['memory_size'])

        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config['batch_size']

        self.reward_scale = config.get('reward_scale', 50.0)

        self.max_queue = 10.0
        self.max_wait = 50.0
        self.max_speed = 1.0

    def _derive_rl_action_counts(self, tls_phases):
        counts = {}
        for tl, phases in tls_phases.items():
            rl_indices = [1 for s in phases if any(c in ("g", "G") for c in s)]
            if sum(rl_indices) == 0:
                counts[tl] = len(phases)
            else:
                counts[tl] = sum(rl_indices)
        return counts

    def normalize_state(self, tls_state, tl):
        q = np.array(tls_state[0:4]) / self.max_queue
        wait = np.array(tls_state[4:8]) / self.max_wait
        speed = np.array(tls_state[8:12]) / self.max_speed
        
        phase_val = tls_state[12]
        action_count = max(1, self.rl_action_counts.get(tl, 1))
        phase = np.array([phase_val / float(action_count)])
        time_in_phase = np.array([tls_state[13] / 50.0])

        return np.concatenate([q, wait, speed, phase, time_in_phase]).astype(np.float32)

    def select_action(self, state, tls, phases):
        action = {}
        for tl in tls:
            action_space = self.rl_action_counts.get(tl, len(phases.get(tl, [])))
            if random.random() < self.epsilon:
                action[tl] = int(np.random.randint(action_space))
            else:
                s_tl = state[tl]
                state_tensor = torch.FloatTensor(self.normalize_state(s_tl, tl)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.policy_net[tl](state_tensor)
                action_idx = torch.argmax(q_values[0][:action_space]).item()
                action[tl] = int(action_idx)
        return action

    def remember(self, tl, state, action, reward, next_state, done):
        scaled_reward = float(reward) / max(1.0, float(self.reward_scale))
        self.memory[tl].append((state, action, scaled_reward, next_state, float(done)))

    def replay(self):
        for tl in list(self.policy_net.keys()):
            if len(self.memory[tl]) < self.batch_size:
                continue

            batch = random.sample(self.memory[tl], self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states_np = np.array([self.normalize_state(s, tl) for s in states], dtype=np.float32)
            states_tensor = torch.from_numpy(states_np).to(self.device)

            next_states_np = np.array([self.normalize_state(s, tl) for s in next_states], dtype=np.float32)
            next_states_tensor = torch.from_numpy(next_states_np).to(self.device)

            actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            dones_tensor = torch.FloatTensor(dones).to(self.device)

            q_values = self.policy_net[tl](states_tensor).gather(1, actions_tensor).squeeze()

            with torch.no_grad():
                max_next_q = self.target_net[tl](next_states_tensor).max(1)[0]
                target_q = rewards_tensor + self.gamma * max_next_q * (1.0 - dones_tensor)

            target_q = target_q.to(q_values.dtype)

            loss_fn = nn.SmoothL1Loss()
            loss = loss_fn(q_values, target_q)

            self.optimizer[tl].zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.policy_net[tl].parameters(), max_norm=10.0)

            self.optimizer[tl].step()

    def update_target(self):
        for tl in self.policy_net.keys():
            self.target_net[tl].load_state_dict(self.policy_net[tl].state_dict())
