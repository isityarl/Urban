import warnings
import numpy as np
from alpyne.constants import EngineState
from alpyne.sim import AnyLogicSim
from alpyne.utils import next_num
import DQNAgent

ACTION_DIMENSION = 30
TRAFFIC_LIGHT_INPUTS = ['Phase'] + [f'Phase{i}' for i in range(1, ACTION_DIMENSION)]
SIM_STOP_TIME = 3600  # seconds
JAVA_PATH = '/usr/lib/jvm/java-21-openjdk/bin/java'
MODEL_PATH = 'rl_test1/Almaty.zip'

def map_action_to_dict(action_array: np.ndarray):
    return {name: bool(action_array[i]) for i, name in enumerate(TRAFFIC_LIGHT_INPUTS)}

class TrafficEnv:
    def __init__(self):
        warnings.filterwarnings("ignore")
        self.sim = AnyLogicSim(
            MODEL_PATH,
            java_path=JAVA_PATH,
            auto_finish=False,
            timeout_initial_connection=120000,
            max_server_await_time=120000,
            sim_params={'engine_overrides': {'stop_time': 360000, 'seed': next_num()}},
            config_defaults={}
        )
        self.action_dim = ACTION_DIMENSION
        self.state_dim = len(self.sim.reset().observation)

    def reset(self):
        status = self.sim.reset()
        obs = np.array(list(status.observation.values()), dtype=np.float32)
        return obs

    def step(self, action_idx):
        action_array = np.zeros(self.action_dim)
        action_array[action_idx] = 1  # one-hot action
        action_payload = map_action_to_dict(action_array)
        status = self.sim.take_action(**action_payload)
        obs = np.array(list(status.observation.values()), dtype=np.float32)
        reward = getattr(status, "reward", 0.0) or 0.0
        done = status.state == EngineState.FINISHED
        return obs, reward, done


env = TrafficEnv()
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

num_episodes = 100
max_steps = 1000

for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        agent.memory.push(state, action, reward, next_state, done)
        agent.update(batch_size=64)

        state = next_state
        total_reward += reward
        if done:
            break

    # update target network every episode
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}")
