import torch
import pandas as pd
import random
import numpy as np
from back.agents.CORE_agent import BaseAgent
from back.RL_env.env import SumoEnv
from back.train.config import config


def eval_DQN(model_path, eval_episodes=5):
    print(f"Model: {model_path}")
    print(f"Episodes: {eval_episodes}")

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    env = SumoEnv(
        cfg_path="back/data/osm.sumocfg",
        net_path="back/data/osm.net.xml.gz",
        gui=True,
        step_length=1
    )

    state = env.reset()
    tls_phases = env.main_phases
    tls_list = list(tls_phases.keys())

    state_size = len(next(iter(state.values())))

    agent = BaseAgent(
        state_size=14,
        tls_phases=tls_phases,
        config=config
    )

    checkpoint = torch.load(model_path, map_location=agent.device)

    for tl, net in agent.policy_net.items():
        if tl in checkpoint["models"]:
            net.load_state_dict(checkpoint["models"][tl])
            net.eval()

    agent.epsilon = 0.0

    tls_list = list(tls_phases.keys())
    action_interval = config.get("action_interval", 15)

    results = []

    for ep in range(eval_episodes):
        print(f"\nEvaluation Episode {ep+1}/{eval_episodes}")

        state = env.reset()
        done = False
        last_action = None
        steps = 0

        shaped_reward = 0.0
        env_reward = 0.0
        acc_rewards = {tl: 0.0 for tl in tls_list}

        while not done:
            if last_action is None or steps % action_interval == 0:
                action = agent.select_action(state, tls_list, tls_phases)
                last_action = action
            else:
                action = last_action

            next_state, rewards, done, _ = env.step(action)
            env_reward += sum(rewards.values())

            for tl in tls_list:
                acc_rewards[tl] += rewards.get(tl, 0.0)

            if steps % action_interval == 0:
                shaped_reward += sum(acc_rewards.values())
                acc_rewards = {tl: 0.0 for tl in tls_list}

            state = next_state
            steps += 1

        print(f"Shaped reward (training-style): {shaped_reward:.2f}")
        print(f"Env reward (raw SUMO):          {env_reward:.2f}")

    env.close()

if __name__ == "__main__":
    eval_DQN(
        model_path="back/res/models/DQN/model_parallel_ep2500.pth",eval_episodes=1)
