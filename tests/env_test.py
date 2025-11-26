import sys
import time
from RL_env.env import SumoEnv
import traci
import numpy as np

env = SumoEnv(
    cfg_path="data/osm.sumocfg",
    net_path="data/osm.net.xml.gz",
    gui=False
)

tls = env.controlled_tls
tl_phases = env.phases
print(tls)
print(tl_phases)

print("Phases")
for tl in tls:
    print(f"\nTLS: {tl}")
    if tl in tl_phases:
        for i, phase in enumerate(tl_phases[tl]):
            print(f"  Phase {i}: {phase}")
    else:
        print("No phases!")


print("\nReset")
state = env.reset()
print("STATE LEN =", len(state))
print("STATE PREVIEW =", state[:30])

print("\nStep")

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

tls_list = [tl for tl in env.controlled_tls if tl in env.phases]

for i in range(3000):
    actions = {tl: np.random.randint(len(env.phases[tl])) for tl in tls_list}
    state, reward, done, info = env.step(actions)
    
    if i % 20 == 0:
        print(f"Step {i}: reward={reward}, state_mean={np.mean(state):.4f}")
    
    if done:
        print("Done, resetting")
        env.reset()



print("REWARD =", reward)
print("DONE =", done)
