from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel

from back.app.services.edge_matcher import EdgeMatcher
from back.app.utils.trips_generator import generate_trip_xml, append_trip_to_file
from back.app.services.sumo_runner import run_sumo
from back.app.services.result_parser import parse_tripinfo
from back.app.utils.capture import run_sumo_capture, parse_sumo_stdout, parse_stats_file, parse_for_special
from back.app.config import TRIPS_FILE, SUMO_CONFIG
from back.train.eval import eval_DQN
from back.train.ppo_eval import evaluate_trained_ppo_single_env
from back.train.basic import evaluate_fixed_time_baseline


matcher = EdgeMatcher("back/data/almaty_edges.geojson")

app = FastAPI(title="UrbanIQ Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class CoordRequest(BaseModel):
    from_coord: list  # [lat, lon]
    to_coord: list

@app.post("/simulate_compare")
def simulate_compare(req: CoordRequest):
    from_edge = matcher.nearest_edge(req.from_coord[0], req.from_coord[1])
    to_edge = matcher.nearest_edge(req.to_coord[0], req.to_coord[1])

    append_trip_to_file(from_edge, to_edge, TRIPS_FILE)

    results = {}
    
    dqn("/home/yarl/Desktop/git/Urban/back/res/models/DQN/model_parallel_final.pth")
    results["DQN_all"] = parse_stats_file()
    results["DQN_spec"] = parse_for_special(veh_id='specialveh')
    ppo("/home/yarl/Desktop/git/Urban/back/res/models/PPO/model_final.pth")
    results["PPO_all"] = parse_stats_file()
    results["PPO_spec"] = parse_for_special(veh_id='specialveh')
    base()
    results["fixed_all"] = parse_stats_file()
    results["fixed_spec"] = parse_for_special(veh_id='specialveh')

    return {
        "from_edge": from_edge,
        "to_edge": to_edge,
        "results": results
    }

app.mount(
    "/ui",
    StaticFiles(directory="front", html=True),
    name="front"
)

def dqn(model_path):
    return eval_DQN(model_path, eval_episodes=1)

def ppo(model_path):
    return evaluate_trained_ppo_single_env(model_path, eval_episodes=1, max_steps=1000, gui=False)

def base():
    return evaluate_fixed_time_baseline(eval_episodes=1, max_steps=1000)
