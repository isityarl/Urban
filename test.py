import warnings
from alpyne.constants import EngineState
from alpyne.sim import AnyLogicSim

warnings.filterwarnings('ignore')

MODEL_PATH = 'rl_test1/Almaty.zip'
JAVA_PATH = '/usr/lib/jvm/java-21-openjdk/bin/java'

# Use only correct keys and float values for RLExperiment
def build_action():
    return {
        "Phase": 0.0,
        "Phase1": 1.0,
        "Phase2": 2.0
    }

sim = AnyLogicSim(
    MODEL_PATH,
    java_path=JAVA_PATH,
    auto_finish=False,
    auto_lock=True,
    timeout_initial_connection=3600,
    max_server_await_time=3600,
)

status = sim.reset()
print("Reset status:")
print("state:", status.state)
print("stop flag:", getattr(status, "stop", None))

obs = status.observation
print("Observation keys after reset:", list(obs.keys()))
try:
    print("Observation values after reset:", dict(obs))
except Exception as e:
    print("Error reading observation after reset:", e)

if "currentTime" in obs:
    print("Current simulation time after reset:", obs["currentTime"])
else:
    print("Simulation time not found in observation.")

if status.state == EngineState.FINISHED:
    raise SystemExit("Simulation finished immediately after reset.")

action = build_action()
print("Action being sent:", action)
status = sim.take_action(_action=action)

print("=== POST ACTION STATUS ===")
print("state:", status.state)
print("stop flag:", getattr(status, "stop", None))

obs = status.observation
print("Observation keys after action:", list(obs.keys()))
try:
    print("Observation values after action:", dict(obs))
except Exception as e:
    print("Error reading observation after action:", e)

# Optional: Print simulation time if present in observation
if "currentTime" in obs:
    print("Current simulation time after action:", obs["currentTime"])
else:
    print("Simulation time not found in observation.")
