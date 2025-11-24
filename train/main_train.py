import argparse
import torch
import random
import numpy as np
from collections import deque
from RL_env.env import SumoEnv
from agents.CORE_agent import BaseAgent
from train.config import config
import pandas as pd
import os


def train(gui, episodes):
    env = SumoEnv(cfg_path='data/osm.sumocfg', net_path='data/osm.net.xml.gz', gui=gui, step_length=1)
    temp_state = env.reset()
    state_size = len(temp_state)
    controlled_tls = env.controlled_tls
    phases = env.phases
    action_size = len(phases[controlled_tls[0]])

    agent = BaseAgent(state_size, action_size, config)
    rewards_per_episode = []

    steps_done = 0
    tls_list = [tl for tl in env.controlled_tls if tl in env.phases]

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, tls_list, env.phases)
            next_state, reward, done, _ = env.step(action)

            for tls, act in action.items():
                agent.remember(state, act, reward, next_state, done)

            agent.replay()

            state = next_state
            episode_reward += reward
            steps_done += 1

            if steps_done % config['target_update_freq'] == 0:
                 agent.update_target()

        rewards_per_episode.append(episode_reward)
        print(f"Episode {episode+1}/{episodes} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()

    torch.save(agent.policy_net.state_dict(), "res/models/DQN/model1.pth")
    print('model saved')    
    
    pd.DataFrame(rewards_per_episode, columns=["reward"]).to_csv("res/logs/DQN/model1.csv", index=False)
    print('rewards saved')
if __name__ == "__main__":
    train(gui=True, episodes=1000)