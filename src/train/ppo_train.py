import torch
import numpy as np
import os
import time
import pandas as pd
from src.agents.PPO_agent import PPOAgent
from src.RL_env.parallel_env import ParallelEnvs
from src.train.config import config

def train_parallel_ppo(num_envs=None, episodes=None, pre_model=None, gui=None):
    print(f"Starting parallel PPO with {num_envs} SUMO environments (CPU/GPU)")

    envs = ParallelEnvs(
        num_envs=num_envs,
        cfg="src/data/osm.sumocfg",
        net="src/data/osm.net.xml.gz",
        gui=gui,
        step_length=1
    )

    states_list = envs.reset()
    first_reset = states_list[0]

    if "main_phases" not in first_reset:
        tls_phases = first_reset.get("phases") or first_reset.get("state_phases") or {}
        print("Warning: 'main_phases' not found. Using fallback.")
    else:
        tls_phases = first_reset["main_phases"]

    agent = PPOAgent(state_size=14, tls_phases=tls_phases, config=config)
    tls_list = list(agent.tls_phases.keys())

    if pre_model is not None and os.path.exists(pre_model):
        checkpoint = torch.load(pre_model, map_location=agent.device)
        for tl, net in agent.policies.items():
            if tl in checkpoint:
                net.load_state_dict(checkpoint[tl])
        print('Pretrained model loaded')

    rewards_all_episodes = []

    rollout_steps = config.get('rollout_steps')
    action_interval = config.get('action_interval')

    for episode in range(1, episodes + 1):
        states_list = envs.reset()

        episode_rewards = [0.0] * num_envs
        done_flags = [False] * num_envs
        last_actions = [{tl: 0 for tl in tls_list} for _ in range(num_envs)]
        last_infos = [{tl: {'state': states_list[i]["state"][tl], 'action': 0, 'logp': 0.0, 'value': 0.0} 
                       for tl in tls_list} for i in range(num_envs)]

        steps_done = 0
        rollout_counter = 0
        t0 = time.time()

        while not all(done_flags):
            actions_list = []
            infos_list = []

            for i, state_dict in enumerate(states_list):
                if last_actions[i] is None or steps_done % action_interval == 0:
                    actions, infos = agent.select_action(state_dict["state"], tls_list)
                    last_actions[i] = actions
                    last_infos[i] = infos
                else:
                    actions = last_actions[i]
                    infos = last_infos[i]

                actions_list.append(actions)
                infos_list.append(infos)

            next_list = envs.step(actions_list)

            for i, s_dict in enumerate(next_list):
                agent.store(infos_list[i], s_dict["rewards"], s_dict["done"])
                episode_rewards[i] += sum(s_dict["rewards"][tl] for tl in tls_list)
                done_flags[i] = done_flags[i] or s_dict["done"]

            states_list = next_list
            steps_done += 1
            rollout_counter += 1

            if rollout_counter >= rollout_steps or all(done_flags):
                last_states_dict = {tl: states_list[0]["state"][tl] for tl in tls_list} if not all(done_flags) else None
                stats = agent.update(last_states=last_states_dict, done=all(done_flags), verbose=True)
                agent.clear_buffers()
                rollout_counter = 0
                if stats:
                    mean_policy = np.mean([s['policy_loss'] for s in stats.values()])
                    mean_value  = np.mean([s['value_loss'] for s in stats.values()])
                    mean_entropy = np.mean([s['entropy'] for s in stats.values()])
                    tot_samples = sum([s['samples'] for s in stats.values()])
                    print(f"Update: samples={tot_samples} | policy_loss={mean_policy:.4f} "
                          f"| value_loss={mean_value:.4f} | entropy={mean_entropy:.4f}")

        ep_time = time.time() - t0
        print(f"Episode {episode}/{episodes} | Rewards: {episode_rewards} | Steps: {steps_done} | Time: {ep_time:.1f}s")
        rewards_all_episodes.append(episode_rewards)

        if episode % 100 == 0:
            ckpt_dir = "src/res/models/PPO"
            os.makedirs(ckpt_dir, exist_ok=True)
            save_path = os.path.join(ckpt_dir, f"model_parallel_ep{episode}.pth")
            torch.save({tl: net.state_dict() for tl, net in agent.policies.items()}, save_path)
            print(f"Saved checkpoint: {save_path}")

    os.makedirs("src/res/models/PPO", exist_ok=True)
    final_path = os.path.join("src/res/models/PPO", "model_parallel_final.pth")
    torch.save({tl: net.state_dict() for tl, net in agent.policies.items()}, final_path)
    print("Final model saved")

    os.makedirs("src/res/logs/PPO", exist_ok=True)
    df = pd.DataFrame(rewards_all_episodes, columns=[f"env{i}" for i in range(num_envs)])
    df.to_csv("src/res/logs/PPO/model_parallel.csv", index=False)
    print("Rewards log saved")

    envs.close()

if __name__ == "__main__":
    train_parallel_ppo(num_envs=4, episodes=2, gui=True)
