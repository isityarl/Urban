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

    agent = BaseAgent(state_size, phases, config)
    rewards_per_episode = []

    tls_list = [tl for tl in env.controlled_tls if tl in env.phases]
    print('Cuda', torch.cuda.is_available())

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps_done = 0
 
        while not done and steps_done < config['max_steps']:
            actions = agent.select_action(state, tls_list, env.phases)
            next_state, reward, done, _ = env.step(actions)

            for tl, action in actions.items():
                agent.remember(tl, state, action, reward, next_state, done)

            agent.replay()

            state = next_state
            episode_reward += reward
            steps_done += 1
            if steps_done % 100 == 0:
                print('Step', steps_done)

            if steps_done % config['target_update_freq'] == 0:
                 agent.update_target()

        rewards_per_episode.append(episode_reward)
        avg_eps = sum(agent.epsilon.values()) / len(agent.epsilon)
        print(f"Episode {episode+1}/{episodes} | Reward: {episode_reward:.2f} | Avg Epsilon: {avg_eps:.3f}")

    env.close()

    torch.save({tl: net.state_dict() for tl, net in agent.policy_net.items()},"res/models/DQN/model1.pth")
    print('model saved')    
    
    pd.DataFrame(rewards_per_episode, columns=["reward"]).to_csv("res/logs/DQN/model1.csv", index=False)
    print('rewards saved')
if __name__ == "__main__":
    train(gui=True, episodes=3)