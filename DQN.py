import numpy as np
import random
from alpyne.constants import EngineState
from alpyne.sim import AnyLogicSim
from alpyne.utils import next_num

# --- MODEL CONSTANTS ---
# The number of traffic lights that can be switched independently
ACTION_DIMENSION = 30 
# The number of metrics observed (e.g., 30 queues + 30 wait times)
OBS_DIMENSION = 60 

# Placeholder names for the 30 boolean inputs of the 'doAction' function in AnyLogic
# This list MUST match the input arguments defined in your AnyLogic 'doAction' exactly.
TRAFFIC_LIGHT_INPUTS = [f'T{i}_Switch' for i in range(ACTION_DIMENSION)]

def map_action_to_dict(action_array: np.ndarray) -> dict:
    """
    Converts a 30-element MultiBinary array (e.g., from an RL model or random generation) 
    into the dictionary of keyword arguments required by sim.take_action().
    """
    action_dict = {}
    for i, name in enumerate(TRAFFIC_LIGHT_INPUTS):
        # Action input in AnyLogic is expected as a boolean
        action_dict[name] = bool(action_array[i])
    return action_dict

# 1. Initialize the AnyLogicSim
# NOTE: The model path is now passed as the FIRST POSITIONAL argument.
sim = AnyLogicSim(
    r'rl_test1/Almaty.zip',  # <-- FIX: Path passed as required positional argument
    auto_finish=True,  # Sets engine to FINISHED if the model's stop condition is met
    # Override simulation run settings
    engine_overrides=dict(
        stop_time=3600,  # Stop after 1 hour (3600 seconds), matching your 'isTerminal' logic
        seed=next_num 
    ),
    # You can pass AnyLogic parameters here if needed
    config_defaults=dict() 
)

# 2. Define the episode parameters
num_episodes = 5
results = {} # To store congestion data per episode

print(f"Starting {num_episodes} simulation episodes...")

for episode in range(num_episodes):
    print(f"\n--- Running Episode {episode + 1} ---")
    
    # Reset the simulation. Status holds the initial observation, etc.
    # We pass no custom arguments here.
    status = sim.reset()
    
    # Track congestion over time for this episode
    congestion_history = []
    
    # Loop until the model's stop condition ('isTerminal' logic) is met
    while EngineState.FINISHED not in status.state:
        
        # 3. Generate a random action (a 30-element array of 0s and 1s)
        # This replaces the RL agent's prediction step.
        random_action_array = np.random.randint(0, 2, size=ACTION_DIMENSION, dtype=np.int32)
        action_payload = map_action_to_dict(random_action_array)
        
        # 4. Take the action. The model advances by its internal decision frequency (e.g., 10 seconds).
        status = sim.take_action(
            **action_payload # Unpack the dictionary into keyword arguments
        )

        # 5. Extract Observation (State after action) and calculate a metric
        # status.observation is the NumPy array returned by the AnyLogic 'getObservation' function.
        current_obs = status.observation 
        
        # Calculate current total congestion (sum of all metrics)
        total_congestion = np.sum(current_obs) 
        congestion_history.append(total_congestion)

        print(f"Time: {status.time:.0f}s, State: {status.state}, Congestion: {total_congestion:.2f}")

    # 6. Store and visualize the results
    # The 'status.stop' boolean indicates if the model finished due to the stop condition
    results[episode] = {
        'congestion_mean': np.mean(congestion_history), 
        'stop_condition_met': status.stop
    }
    
print("\n--- Summary ---")
for ep, res in results.items():
    print(f"Episode {ep + 1}: Avg Congestion = {res['congestion_mean']:.2f}, Terminated? {res['stop_condition_met']}")

# Always close the connection
sim.close()
