import warnings
import numpy as np
from alpyne.constants import EngineState
from alpyne.sim import AnyLogicSim
from alpyne.utils import next_num

warnings.filterwarnings('ignore')

MODEL_PATH = 'rl_test1/Almaty.zip'
JAVA_PATH = '/usr/lib/jvm/java-21-openjdk/bin/java'

def map_action_to_dict(action_array):
    keys = ['Phase'] + [f'Phase{i}' for i in range(1, 30)]
    return {k: bool(action_array[i]) for i, k in enumerate(keys)}

print("Starting AnyLogic Simulation Server (diagnostic)...")
sim = AnyLogicSim(
    MODEL_PATH,
    java_path=JAVA_PATH,
    auto_finish=False,
    timeout_initial_connection=120000,
    max_server_await_time=120000,
    sim_params={'engine_overrides': {'stop_time': 360000, 'seed': next_num()}},
    config_defaults={}
)

status = sim.reset()
print("=== RESET STATUS ===")
print("state:", status.state)
print("time_in_model:", getattr(status, "time_in_model", None))
print("progress:", getattr(status, "progress", None))
print("stop flag:", getattr(status, "stop", None))
print("message:", getattr(status, "message", None))
try:
    print("observation (mapping):", status.observation)
    print("observation dict:", dict(status.observation))
except Exception as e:
    print("Error reading observation:", e)

if status.state == EngineState.FINISHED:
    print("Simulation already FINISHED immediately after reset. Exiting diagnostic.")
    raise SystemExit("Finished after reset - check model stop time or isTerminal()")
