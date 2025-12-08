import argparse
import torch
import random
import numpy as np
from collections import deque
from src.RL_env.env import SumoEnv
from src.agents.CORE_agent import BaseAgent
from src.train.config import config
import pandas as pd
import os


def train(gui, episodes, pre_model=None):
    env = SumoEnv(cfg_path='src/data/osm.sumocfg', net_path='src/data/osm.net.xml.gz', gui=gui, step_length=1)
    state_size = 14

    agent = BaseAgent(state_size, env.main_phases, config)

    if pre_model is not None and os.path.exists(pre_model):
        checkpoint = torch.load(pre_model, map_location=agent.device)
        for tl, net in agent.policy_net.items():
            if tl in checkpoint:
                net.load_state_dict(checkpoint[tl])

        agent.update_target()
        print('Model loaded')

    rewards_per_episode = []
    tls_list = env.main_tls
    print('Cuda', torch.cuda.is_available())

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps_done = 0

        interval = 15
        while not done and steps_done < config['max_steps']:
            if steps_done % interval == 0:
                actions = agent.select_action(state, tls_list, env.main_phases)
                last_actions = actions
            else:
                actions = last_actions    
            next_state, rewards, done, _ = env.step(actions)

            for tl, action in actions.items():
                s_tl = state[tl]
                ns_tl = next_state[tl]
                r_tl = rewards[tl]
                agent.remember(tl, s_tl, action, r_tl, ns_tl, done)
                episode_reward += r_tl

            agent.replay()

            state = next_state
            steps_done += 1
            if steps_done % 100 == 0:
                print('Step', steps_done)

            if steps_done % config['target_update_freq'] == 0:
                 agent.update_target()

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        print(f"Episode {episode+1}/{episodes} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
        rewards_per_episode.append(episode_reward)

    env.close()

    torch.save({tl: net.state_dict() for tl, net in agent.policy_net.items()},"src/res/models/DQN/model1.pth")
    print('model saved')    
    
    pd.DataFrame(rewards_per_episode, columns=["reward"]).to_csv("src/res/logs/DQN/model1.csv", index=False)
    print('rewards saved')
if __name__ == "__main__":
    train(gui=True, episodes=1, pre_model='src/res/models/DQN/model1.pth')