import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import pandas as pd
from agents.DQN import DQN

class BaseAgent:
    def __init__(self, state_size, tls_phases, config):
        self.state_size = state_size
        self.tls_phases = tls_phases
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = {}
        self.target_net = {}
        self.optimizer = {}
        self.memory = {}
        self.epsilon = {}

        for tl, phases in tls_phases.items():
            action_size = len(phases)
            self.policy_net[tl] = DQN(state_size, action_size).to(self.device)
            self.target_net[tl] = DQN(state_size, action_size).to(self.device)
            self.target_net[tl].load_state_dict(self.policy_net[tl].state_dict())
            self.target_net[tl].eval()
            self.optimizer[tl] = optim.AdamW(self.policy_net[tl].parameters(), lr=config['learning_rate'])
            self.memory[tl] = deque(maxlen=config['memory_size'])
            self.epsilon[tl] = config['epsilon_start']

        self.gamma = config['gamma']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.batch_size = config['batch_size']

    def build_model(self):
        return DQN(self.state_size, self.action_size)

    def select_action(self, state, tls, phases):
        action = {}
        for tl in tls:
            if random.random() < self.epsilon[tl]:
                action[tl] = np.random.randint(len(phases[tl]))
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.policy_net[tl](state_tensor)
                
                action[tl] = torch.argmax(q_values[0][:len(phases[tl])]).item()
        return action        
    
    def remember(self, tl, state, action, reward, next_state, done):
        self.memory[tl].append((state, action, reward, next_state, done))


    def replay(self):
        for tl in self.policy_net.keys():
            if len(self.memory[tl]) < self.batch_size:
                return

            batch = random.sample(self.memory[tl], self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            q_values = self.policy_net[tl](states).gather(1, actions).squeeze()
            with torch.no_grad():
                max_next_q = self.target_net[tl](next_states).max(1)[0]
                target_q = rewards + self.gamma * max_next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q)
            self.optimizer[tl].zero_grad()
            loss.backward()
            self.optimizer[tl].step()

            if self.epsilon[tl] > self.epsilon_min:
                self.epsilon[tl] *= self.epsilon_decay

    def update_target(self):
        for tl in self.policy_net.keys():
            self.target_net[tl].load_state_dict(self.policy_net[tl].state_dict())