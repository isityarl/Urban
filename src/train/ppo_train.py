# ppo_train.py
import torch
from src.RL_env.env import SumoEnv
from src.agents.PPO_agent import PPOAgent
from src.train.config import config
import pandas as pd
import numpy as np
import os
import time

def train_ppo(gui, episodes, pre_model=None):
    env = SumoEnv(cfg_path='src/data/osm.sumocfg', net_path='src/data/osm.net.xml.gz', gui=gui, step_length=1)
    state_size = 14

    agent = PPOAgent(state_size, env.main_phases, config)

    if pre_model is not None and os.path.exists(pre_model):
        checkpoint = torch.load(pre_model, map_location=agent.device)
        for tl, net in agent.policies.items():
            if tl in checkpoint:
                net.load_state_dict(checkpoint[tl])
        print('Model loaded')

    rewards_per_episode = []
    tls_list = env.main_tls
    print('Cuda', torch.cuda.is_available())

    rollout_steps = config.get('rollout_steps', 1024)
    interval = config.get('action_repeat', 15)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        steps_done = 0
        done = False
        last_actions = {tl: 0 for tl in tls_list}
        last_infos = {tl: {'state': state[tl], 'action': 0, 'logp': 0.0, 'value': 0.0} for tl in tls_list}

        rollout_counter = 0
        t0 = time.time()
        while not done and steps_done < config['max_steps']:
            if steps_done % interval == 0:
                actions, infos = agent.select_action(state, tls_list)
                last_actions = actions
                last_infos = infos
            else:
                actions = last_actions
                infos = last_infos

            next_state, rewards, done, _ = env.step(actions)

            agent.store(infos, rewards, done)

            episode_reward += sum(rewards[tl] for tl in tls_list)
            state = next_state
            steps_done += 1
            rollout_counter += 1

            if rollout_counter >= rollout_steps or done:
                last_states = state if not done else None
                stats = agent.update(last_states=last_states, done=done, verbose=True)
                agent.clear_buffers()
                rollout_counter = 0
                if stats:
                    mean_policy = []
                    mean_value = []
                    mean_entropy = []
                    tot_samples = 0
                    for tl, s in stats.items():
                        mean_policy.append(s['policy_loss'])
                        mean_value.append(s['value_loss'])
                        mean_entropy.append(s['entropy'])
                        tot_samples += s['samples']
                    print(f"Update: samples={tot_samples} | policy_loss={np.mean(mean_policy):.4f} "
                          f"| value_loss={np.mean(mean_value):.4f} | entropy={np.mean(mean_entropy):.4f}")

        ep_time = time.time() - t0
        print(f"Episode {episode+1}/{episodes} | Reward: {episode_reward:.2f} | Steps: {steps_done} | Time: {ep_time:.1f}s")
        rewards_per_episode.append(episode_reward)

    env.close()

    save_path = "src/res/models/PPO/model1.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({tl: net.state_dict() for tl, net in agent.policies.items()}, save_path)
    print('model saved')

    os.makedirs("src/res/logs/PPO", exist_ok=True)
    pd.DataFrame(rewards_per_episode, columns=["reward"]).to_csv("src/res/logs/PPO/model1.csv", index=False)


if __name__ == "__main__":
    train_ppo(gui=True, episodes=1)
