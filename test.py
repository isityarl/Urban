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

sim = AnyLogicSim(
    MODEL_PATH,
    java_path=JAVA_PATH,
    auto_finish=False,
    auto_lock=True,
    timeout_initial_connection=120000,
    max_server_await_time=120000,
)

status = sim.reset()

print("Reset status:")
print("state:", status.state)
print("stop flag:", getattr(status, "stop", None))

if status.state == EngineState.FINISHED:
    raise SystemExit("Simulation finished immediately after reset.")

dummy_action = {k: False for k in ['Phase'] + [f'Phase{i}' for i in range(1, 30)]}
status = sim.take_action(_action=dummy_action)

print("=== POST ACTION STATUS ===")
print("state:", status.state)
print("stop flag:", getattr(status, "stop", None))

# --- 3. Query observations ---
obs = status.observation
print("Observation keys:", list(obs.keys()))
try:
    print("Observation values:", dict(obs))
except Exception as e:
    print("Error reading observation:", e)
