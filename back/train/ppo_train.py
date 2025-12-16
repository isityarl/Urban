import torch
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from back.agents.PPO_agent import PPOAgent
from back.RL_env.env import SumoEnv
from back.train.config import config

def train_ppo(episodes=100, pre_model=None, gui=False, live_plot=True):
    env = SumoEnv(
        cfg_path="back/data/osm.sumocfg",
        net_path="back/data/osm.net.xml.gz",
        gui=gui,
        step_length=1
    )

    tls_phases = env.main_phases
    tls_list = list(tls_phases.keys())

    agent = PPOAgent(state_size=14, tls_phases=tls_phases, config=config)

    if pre_model is not None and os.path.exists(pre_model):
        ckpt = torch.load(pre_model, map_location=agent.device)
        agent.body.load_state_dict(ckpt["body"])
        agent.heads.load_state_dict(ckpt["heads"])
        if "optimizer" in ckpt:
            agent.optimizer.load_state_dict(ckpt["optimizer"])
        start_episode = ckpt.get("episode", 0) + 1
        print(f"Continuing PPO training from episode {start_episode}")
    else:
        start_episode = 1

    action_interval = config.get("action_interval", 15)
    rollout_steps = config.get("rollout_steps", 2048)

    rewards_log = []

    for episode in range(start_episode, episodes + 1):
        state = env.reset(episode=episode)
        done = False
        ep_reward = 0
        step_count = 0
        rollout_count = 0
        last_action = None
        last_infos = None
        acc_rewards = {tl: 0.0 for tl in tls_list}
        t0 = time.time()

        while not done:
            if last_action is None or step_count % action_interval == 0:
                actions, infos = agent.select_action(state, tls_list)
                last_action = actions
                last_infos = infos
            else:
                actions = last_action
                infos = last_infos

            next_state, rewards, done, _ = env.step(actions)

            for tl in tls_list:
                acc_rewards[tl] += rewards[tl]
            ep_reward += sum(rewards.values())

            step_count += 1
            if step_count % action_interval == 0:
                agent.store(last_infos, acc_rewards, done, time_limit=(step_count >= env.max_steps))
                acc_rewards = {tl: 0.0 for tl in tls_list}
                rollout_count += 1

            state = next_state    

            if rollout_count >= rollout_steps or done:
                stats = agent.update(last_states=state if not done else None, done=done, verbose=True)
                agent.clear_buffers()
                rollout_count = 0

                if stats:
                    tl_stats = {k: v for k, v in stats.items() if isinstance(v, dict)}
                    mean_grad = stats.get("grad_norm", 0.0)
                    
                    if tl_stats:
                        mean_pol = np.mean([s["policy_loss"] for s in tl_stats.values()])
                        mean_val = np.mean([s["value_loss"] for s in tl_stats.values()])
                        mean_ent = np.mean([s["entropy"] for s in tl_stats.values()])
                        samples = sum(s["samples"] for s in tl_stats.values())
                    else:
                        mean_pol = mean_val = mean_ent = 0.0
                        samples = 0

                    print(
                        f"PPO update | samples={samples} "
                        f"| policy={mean_pol:.4f} "
                        f"| value={mean_val:.4f} "
                        f"| entropy={mean_ent:.4f} "
                        f"| grad_norm={mean_grad:.4f}"
                    )

        ep_time = time.time() - t0
        rewards_log.append(ep_reward)

        print(
            f"Episode {episode}/{episodes} | "
            f"Reward: {ep_reward:.2f} | "
            f"Steps: {step_count} | "
            f"Time: {ep_time:.1f}s"
        )

        if episode % 100 == 0:
            os.makedirs('back/res/models/PPO', exist_ok=True)
            save_path = f"back/res/models/PPO/model_ep{episode}.pth"
            torch.save({
                "body": agent.body.state_dict(),
                "heads": agent.heads.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "episode": episode
            }, save_path)
            print("Saved checkpoint")

            reward_path = f"back/res/logs/PPO/training_rewards_ep{episode}.csv"
            pd.DataFrame({"reward": rewards_log}).to_csv(reward_path, index=False)
            print("Saved rewards")

    os.makedirs("back/res/models/PPO", exist_ok=True)
    final_path = "back/res/models/PPO/model_final.pth"
    torch.save(
        {
            "body": agent.body.state_dict(),
            "heads": agent.heads.state_dict(),
        },
        final_path
    )
    print("Final model saved")

    os.makedirs("back/res/logs/PPO", exist_ok=True)
    pd.DataFrame({"reward": rewards_log}).to_csv(
        "back/res/logs/PPO/training_rewards.csv",
        index=False
    )
    print("Reward log saved")

    env.close()

    if live_plot:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    train_ppo(episodes=2, gui=False)
