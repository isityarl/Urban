import torch
import pandas as pd
from back.agents.PPO_agent import PPOAgent
from back.RL_env.env import SumoEnv
from back.train.config import config


def evaluate_trained_ppo_single_env(model_path, eval_episodes=1, max_steps=1000, gui=True):
    env = SumoEnv(
        cfg_path="back/data/osm.sumocfg",
        net_path="back/data/osm.net.xml.gz",
        gui=gui,
        step_length=1
    )

    state = env.reset()
    tls_phases = env.main_phases
    tls_list = list(tls_phases.keys())

    agent = PPOAgent(
        state_size=14,
        tls_phases=tls_phases,
        config=config
    )

    checkpoint = torch.load(model_path, map_location=agent.device)
    agent.body.load_state_dict(checkpoint["body"])
    agent.heads.load_state_dict(checkpoint["heads"])

    agent.body.eval()
    agent.heads.eval()

    action_interval = config.get("action_interval", 15)

    results = []

    for ep in range(eval_episodes):
        print(f"\nEvaluation Episode {ep+1}/{eval_episodes}")

        state = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        last_actions = None

        while not done and steps < max_steps:

            if last_actions is None or steps % action_interval == 0:
                with torch.no_grad():
                    actions, _ = agent.select_action(
                        state,
                        tls_list,
                        deterministic=True 
                    )
                last_actions = actions
            else:
                actions = last_actions

            state, rewards, done, _ = env.step(actions)

            total_reward += sum(rewards.values())
            steps += 1

        print(f"Episode reward: {total_reward:.2f} | Steps: {steps}")

    env.close()

if __name__ == "__main__":
    evaluate_trained_ppo_single_env(
        model_path="back/res/models/PPO/model_ep1300.pth",
        eval_episodes=1,
        max_steps=1000,
        gui=False
    )