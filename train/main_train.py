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


def train(gui=False, episodes=500):
    env = SumoEnv(cfg_path='data/osm.sumocfg', net_path='data/network.net.xml', gui=gui, step_length=1)
    state_size = len(env.get_state())
    action_size = len(env.phases[env.controlled_tls[0]])

    agent = BaseAgent(state_size, action_size, config)
    rewards_per_episode = []

    steps_done = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = {tls: agent.select_action(state) for tls in env.controlled_tls}
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action[env.controlled_tls[0]], reward, next_state, done)
            agent.replay()

            state = next_state
            episode_reward += reward
            steps_done += 1

            if steps_done % config['target_update_freq'] == 0:
                agent.update_target()

        rewards_per_episode.append(episode_reward)
        print(f"Episode {episode+1}/{episodes} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()

    # Сохраняем модель
    torch.save(agent.policy_net.state_dict(), "trained_models/dqn_model.pth")
    print("Model saved to trained_models/dqn_model.pth")

    # Сохраняем reward
    
    os.makedirs("trained_models", exist_ok=True)
    pd.DataFrame(rewards_per_episode, columns=["reward"]).to_csv("trained_models/rewards.csv", index=False)
    print("Rewards saved to trained_models/rewards.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to SUMO .sumocfg file")
    parser.add_argument("--net", required=True, help="Path to SUMO .net.xml file")
    parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train")
    args = parser.parse_args()

    train(args.cfg, args.net, gui=args.gui, episodes=args.episodes)