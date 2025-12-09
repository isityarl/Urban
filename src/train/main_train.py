import torch
import numpy as np
import random
from src.agents.CORE_agent import BaseAgent
from src.RL_env.parallel_env import ParallelEnvs
from src.train.config import config
import pandas as pd
import os

def train_parallel_dqn(num_envs=4, episodes=50):
    print(f"Starting parallel DQN with {num_envs} SUMO environments (CPU only)")

    envs = ParallelEnvs(
        num_envs=num_envs,
        cfg="src/data/osm.sumocfg",
        net="src/data/osm.net.xml.gz",
        gui=False,
        step_length=1
    )

    states_and_info = envs.reset()
    first_reset = states_and_info[0]
    tls_phases = first_reset["main_phases"]
    print("Example TLS phases (env0):", tls_phases)

    states_list = [r["state"] for r in states_and_info]

    agent = BaseAgent(state_size=14, tls_phases=tls_phases, config=config)
    tls_list = list(agent.tls_phases.keys())

    rewards_all_episodes = []

    for episode in range(episodes):
        states_and_info = envs.reset()
        states_list = [r["state"] for r in states_and_info]
       
        episode_rewards = [0.0] * num_envs
        done_flags = [False] * num_envs
        last_actions = [None] * num_envs
        steps_done = 0

        print(f"\n=== Episode {episode+1}/{episodes} ===")

        while not all(done_flags):
            actions_list = []

            for i, state in enumerate(states_list):
                if last_actions[i] is None or steps_done % 15 == 0:
                    action = agent.select_action(state, tls_list, agent.tls_phases)
                    last_actions[i] = action
                else:
                    action = last_actions[i]
                actions_list.append(action)

            next_list = envs.step(actions_list)

            for i, (next_state, rewards, done, _) in enumerate(next_list):
                total_reward = sum(rewards.values())
                episode_rewards[i] += total_reward

                for tl, act in actions_list[i].items():
                    agent.remember(
                        tl,
                        states_list[i][tl],
                        act,
                        rewards[tl],
                        next_state[tl],
                        done
                    )

                if done:
                    done_flags[i] = True

            agent.replay()

            if steps_done % 100 == 0:
                mem_sizes = [len(agent.memory[tl]) for tl in tls_list]
                print(f"Step {steps_done} | Memory sizes per TLS: {mem_sizes}")

            states_list = [n[0] for n in next_list]
            steps_done += 1

            if steps_done % config['target_update_freq'] == 0:
                agent.update_target()
                print(f"Step {steps_done} | Target networks updated")

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        print(f"Episode {episode+1}/{episodes} | Epsilon: {agent.epsilon:.3f}")
        for env_id, r in enumerate(episode_rewards):
            print(f"  Env {env_id}: {r:.2f}")
        rewards_all_episodes.append(episode_rewards)

        print(f"Episode {episode+1} completed | Rewards per env: {episode_rewards} | Epsilon: {agent.epsilon:.3f}")
        print("-" * 60)

    envs.close()

    torch.save({tl: net.state_dict() for tl, net in agent.policy_net.items()},"src/res/models/DQN/model_parallel.pth")
    pd.DataFrame(rewards_all_episodes).to_csv("src/res/logs/DQN/model_parallel.csv", index=False)

    print("\nTraining finished. Models and rewards saved.")

if __name__ == "__main__":
    train_parallel_dqn(num_envs=4, episodes=1)
