import torch
import numpy as np
import random
import os
from src.agents.CORE_agent import BaseAgent
from src.RL_env.parallel_env import ParallelEnvs
from src.train.config import config
import pandas as pd

def train_parallel_dqn(num_envs=4, episodes=50, pre_model=None):
    print(f"Starting parallel DQN with {num_envs} SUMO environments (CPU only)")

    log_dir = "src/res/logs/DQN/details"
    os.makedirs(log_dir, exist_ok=True)

    envs = ParallelEnvs(
        num_envs=num_envs,
        cfg="src/data/osm.sumocfg",
        net="src/data/osm.net.xml.gz",
        gui=False,
        step_length=1
    )

    states_and_info = envs.reset()
    first_reset = states_and_info[0]
    if "main_phases" not in first_reset:
        tls_phases = first_reset.get("phases") or first_reset.get("state_phases") or {}
        print("Warning: 'main_phases' not found in ParallelEnvs.reset() output. Using fallback.")
    else:
        tls_phases = first_reset["main_phases"]

    print("Example TLS phases (env0):", list(tls_phases.keys())[:5])

    rl_action_counts = {}
    for tl, phases in tls_phases.items():
        rl_indices = [1 for s in phases if any(c in ("g", "G") for c in s)]
        rl_action_counts[tl] = sum(rl_indices) if sum(rl_indices) > 0 else len(phases)

    agent = BaseAgent(state_size=14, tls_phases=tls_phases, config=config, rl_action_counts=rl_action_counts)
    tls_list = list(agent.tls_phases.keys())

    if pre_model is not None and os.path.exists(pre_model):
        print(f"\nLoading pretrained model: {pre_model}")

        checkpoint = torch.load(pre_model, map_location=agent.device)

        for tl, net in agent.policy_net.items():
            if tl in checkpoint['models']:
                net.load_state_dict(checkpoint['models'][tl])

        for tl, net in agent.target_net.items():
            if tl in checkpoint['models']:
                net.load_state_dict(checkpoint['models'][tl])

        for tl, opt in agent.optimizer.items():
            if tl in checkpoint['optimizers']:
                opt.load_state_dict(checkpoint['optimizers'][tl])

        agent.epsilon = checkpoint.get('epsilon', agent.epsilon)

        start_episode = checkpoint.get('episode', 0)
        print(f"Resuming training from episode {start_episode+1}")

    else:
        start_episode = 0
        print("No pretrained model provided. Starting from scratch.")


    rewards_all_episodes = []

    for episode in range(start_episode, episodes):
        states_and_info = envs.reset(episode=episode)
        states_list = [r["state"] for r in states_and_info]

        episode_rewards = [0.0] * num_envs
        done_flags = [False] * num_envs
        last_actions = [None] * num_envs
        steps_done = 0

        action_interval = config.get('action_interval', 15)

        print(f"\nEpisode {episode+1}/{episodes}")

        while not all(done_flags):
            actions_list = []

            for i, state in enumerate(states_list):
                if last_actions[i] is None or steps_done % action_interval == 0:
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
                    if tl not in agent.memory:
                        continue
                    agent.remember(
                        tl,
                        states_list[i][tl],
                        act,
                        rewards.get(tl, 0.0),
                        next_state[tl],
                        done
                    )

                if done:
                    done_flags[i] = True

            agent.replay()

            if steps_done % 100 == 0:
                total_mem = sum(len(agent.memory[tl]) for tl in tls_list)
                print(f"Step {steps_done} | Total memory: {total_mem}")

            states_list = [n[0] for n in next_list]
            steps_done += 1

            if steps_done % config.get('target_update_freq', 1000) == 0:
                agent.update_target()
                print(f"Step {steps_done} | Target networks updated")

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        print(f"Episode {episode+1}/{episodes} | Epsilon: {agent.epsilon:.3f}")
        for env_id, r in enumerate(episode_rewards):
            print(f"  Env {env_id}: {r:.2f}")
        rewards_all_episodes.append(episode_rewards)

        if (episode + 1) % 100 == 0:
            ckpt_dir = "src/res/models/DQN"
            os.makedirs(ckpt_dir, exist_ok=True)
            save_path = os.path.join(ckpt_dir, f"model_parallel_ep{episode+1}.pth")
            torch.save({
                'models': {tl: net.state_dict() for tl, net in agent.policy_net.items()},
                'optimizers': {tl: opt.state_dict() for tl, opt in agent.optimizer.items()},
                'epsilon': agent.epsilon,
                'episode': episode+1
            }, save_path)
            print(f"Saved checkpoint at episode {episode+1}: {save_path}")


        print(f"Episode {episode+1} completed | Rewards per env: {episode_rewards} | Epsilon: {agent.epsilon:.3f}")
        print("-" * 60)

    envs.close()

    final_dir = "src/res/models/DQN"
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, "model_parallel_final.pth")
    torch.save({
        'models': {tl: net.state_dict() for tl, net in agent.policy_net.items()},
        'optimizers': {tl: opt.state_dict() for tl, opt in agent.optimizer.items()},
        'epsilon': agent.epsilon
    }, final_path)

    pd.DataFrame(rewards_all_episodes).to_csv("src/res/logs/DQN/model_parallel.csv", index=False)
    print("\nTraining finished. Models and rewards saved.")

if __name__ == "__main__":
    train_parallel_dqn(num_envs=4, episodes=1000, pre_model="src/res/models/DQN/model_parallel_ep300.pth")