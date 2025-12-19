from back.app.utils.capture import (
    run_sumo_capture,
    parse_sumo_stdout,
    parse_stats_file,
    parse_for_special
)
from back.train.eval import eval_DQN
from back.train.ppo_eval import evaluate_trained_ppo_single_env
from back.train.basic import evaluate_fixed_time_baseline


def dqn(model_path):
    return eval_DQN(model_path, eval_episodes=1)


results = {}

dqn(model_path="/home/yarl/Desktop/git/Urban/back/res/models/DQN/model_parallel_final.pth")
results["DQN_all"] = parse_stats_file()
results["DQN_spec"] = parse_for_special(veh_id="specialveh")

print(results)