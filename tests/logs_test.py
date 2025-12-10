# test_env_logs.py
from src.RL_env.env import SumoEnv
import os
import glob

cfg_path = "src/data/osm.sumocfg"
net_path = "src/data/osm.net.xml.gz"

env_id = 0
episode = 0

env = SumoEnv(cfg_path=cfg_path, net_path=net_path, gui=False, step_length=1)
state = env.reset(episode=episode, env_id=env_id)

for _ in range(10):
    actions = {tl: 0 for tl in env.main_tls}
    state, rewards, done, _ = env.step(actions)
    if done:
        break

env.close()

log_pattern = f"src/res/logs/DQN/details/env_{env_id}_ep{episode}.log"
if os.path.exists(log_pattern):
    print("Found log file:", log_pattern)
    with open(log_pattern, "r") as f:
        lines = f.readlines()[-20:]
    print("Last lines of log:")
    for line in lines:
        print(line.rstrip())
else:
    print("No log file found at", log_pattern)
