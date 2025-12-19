from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel
from back.app.services.edge_matcher import EdgeMatcher
from back.app.utils.trips_generator import generate_trip_xml
from back.app.services.sumo_runner import run_sumo
from back.app.services.result_parser import parse_tripinfo
from back.app.config import TRIPS_FILE

matcher = EdgeMatcher("back/data/almaty_edges.geojson")

app = FastAPI(title="UrbanIQ Backend")

class CoordRequest(BaseModel):
    from_coord: list  # [lat, lon]
    to_coord: list

@app.post("/simulate_coords")
def simulate_coords(req: CoordRequest):
    from_edge = matcher.nearest_edge(req.from_coord[0], req.from_coord[1])
    to_edge = matcher.nearest_edge(req.to_coord[0], req.to_coord[1])

    xml = generate_trip_xml(from_edge, to_edge)
    with open(TRIPS_FILE, "w") as f:
        f.write(xml)

    run_sumo()
    return {
        "from_edge": from_edge,
        "to_edge": to_edge,
        "result": parse_tripinfo()
    }

app.mount(
    "/ui",
    StaticFiles(directory="front", html=True),
    name="front"
)