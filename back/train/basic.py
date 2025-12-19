import pandas as pd
from back.RL_env.env import SumoEnv
from back.train.config import config

def evaluate_fixed_time_baseline(
    eval_episodes=1,
    max_steps=1000
):
    print("\n===== FIXED-TIME SUMO BASELINE =====")
    print(f"Episodes: {eval_episodes}")

    env = SumoEnv(
        cfg_path="back/data/osm.sumocfg",
        net_path="back/data/osm.net.xml.gz",
        gui=False,
        step_length=1,
        scale=3
    )

    results = []

    for ep in range(eval_episodes):
        print(f"\nBaseline Episode {ep+1}/{eval_episodes}")

        state = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < max_steps:
            state, rewards, done, _ = env.step(None)

            total_reward += sum(rewards.values())
            steps += 1

        print(f"Episode reward: {total_reward:.2f}")

        results.append({
            "episode": ep,
            "total_reward": total_reward,
            "policy": "fixed_time"
        })

    env.close()

    df = pd.DataFrame(results)
    df.to_csv(
        "back/res/logs/DQN/evaluate/eval_fixed_time_baseline.csv",
        index=False
    )

    print("\n===== BASELINE FINISHED =====")
    print("Rewards saved for fixed-time SUMO baseline.")

if __name__ == "__main__":
    evaluate_fixed_time_baseline(
        eval_episodes=1,
        max_steps=1000
    )
