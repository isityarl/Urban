import torch
import pandas as pd
from back.agents.CORE_agent import BaseAgent
from back.RL_env.env import SumoEnv
from back.train.config import config


def evaluate_trained_model_single_env(
    model_path,
    eval_episodes=5
):
    print("\n===== SINGLE-ENV EVALUATION =====")
    print(f"Model: {model_path}")
    print(f"Episodes: {eval_episodes}")

    # ---- Create SUMO environment ----
    env = SumoEnv(
        cfg_path="back/data/osm.sumocfg",
        net_path="back/data/osm.net.xml.gz",
        gui=True,
        step_length=1
    )

    # ---- Reset once to get TLS phases ----
    state = env.reset()
    tls_phases = env.main_phases

    # ---- Create agent ----
    agent = BaseAgent(
        state_size=14,
        tls_phases=tls_phases,
        config=config
    )

    # ---- Load trained model ----
    checkpoint = torch.load(model_path, map_location=agent.device)

    for tl, net in agent.policy_net.items():
        if tl in checkpoint["models"]:
            net.load_state_dict(checkpoint["models"][tl])
            net.eval()

    # 🔴 Disable exploration
    agent.epsilon = 0.0

    tls_list = list(tls_phases.keys())
    action_interval = config.get("action_interval", 30)

    results = []

    for ep in range(eval_episodes):
        print(f"\nEvaluation Episode {ep+1}/{eval_episodes}")

        state = env.reset()
        done = False
        last_action = None
        steps = 0

        total_reward = 0.0

        while not done:
            if last_action is None or steps % action_interval == 0:
                action = agent.select_action(state, tls_list, tls_phases)
                last_action = action
            else:
                action = last_action

            state, rewards, done, _ = env.step(action)

            total_reward += sum(rewards.values())
            steps += 1

        print(f"Episode reward: {total_reward:.2f}")

        results.append({
            "episode": ep,
            "total_reward": total_reward
        })

    env.close()

    df = pd.DataFrame(results)
    save_path = "back/res/logs/DQN/evaluate/eval_single_env_rewards.csv"
    df.to_csv(save_path, index=False)

    print("\n===== EVALUATION FINISHED =====")
    print(f"Rewards saved to: {save_path}")
    print(
        "\n📌 NOTE:\n"
        "SUMO performance metrics (WaitingTime, TimeLoss, Speed, etc.)\n"
        "are automatically written to the SUMO log file specified in osm.sumocfg.\n"
        "Compare those logs against your baseline runs to show improvement."
    )


if __name__ == "__main__":
    evaluate_trained_model_single_env(
        model_path="back/res/models/DQN/model_parallel_ep1600.pth",
        eval_episodes=1
    )
