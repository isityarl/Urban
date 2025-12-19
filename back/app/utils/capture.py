import subprocess
import xml.etree.ElementTree as ET

def run_sumo_capture(cfg_path="back/data/osm.sumocfg"):
    sumo_cmd = [
        "sumo",
        "-c", cfg_path,
        "--no-step-log",
        "--summary", "true"
    ]
    
    result = subprocess.run(sumo_cmd, capture_output=True, text=True)
    output = result.stdout
    return output

def parse_sumo_stdout(output):
    results = {}
    for line in output.splitlines():
        line = line.strip()
        if "RouteLength:" in line:
            results["RouteLength"] = float(line.split(":")[1].strip())
        elif "Speed:" in line:
            results["Speed"] = float(line.split(":")[1].strip())
        elif "Duration:" in line:
            results["Duration"] = float(line.split(":")[1].strip())
        elif "WaitingTime:" in line:
            results["WaitingTime"] = float(line.split(":")[1].strip())
        elif "TimeLoss:" in line:
            results["TimeLoss"] = float(line.split(":")[1].strip())
        elif "DepartDelay:" in line:
            results["DepartDelay"] = float(line.split(":")[1].strip())
    return results

def parse_stats_file(filename="/home/yarl/Desktop/git/Urban/back/baseline/stats.xml"):
    tree = ET.parse(filename)
    root = tree.getroot()

    vts = root.find("vehicleTripStatistics")
    if vts is None:
        return {}

    stats = {
        "RouteLength": float(vts.get("routeLength", 0.0)),
        "Speed": float(vts.get("speed", 0.0)),
        "Duration": float(vts.get("duration", 0.0)),
        "WaitingTime": float(vts.get("waitingTime", 0.0)),
        "TimeLoss": float(vts.get("timeLoss", 0.0)),
        "DepartDelay": float(vts.get("departDelay", 0.0))
    }
    return stats


def parse_for_special(filename="/home/yarl/Desktop/git/Urban/back/baseline/tripinfos.xml", veh_id="veh0"):
    tree = ET.parse(filename)
    root = tree.getroot()

    for trip in root.findall("tripinfo"):
        if trip.attrib.get("id") == veh_id:
            duration = float(trip.attrib.get("duration", 0.0))
            route_length = float(trip.attrib.get("routeLength", 0.0))
            waiting_time = float(trip.attrib.get("waitingTime", 0.0))
            time_loss = float(trip.attrib.get("timeLoss", 0.0))
            depart_delay = float(trip.attrib.get("departDelay", 0.0))
            speed = route_length / duration if duration > 0 else 0.0

            stats = {
                "RouteLength": route_length,
                "Speed": speed,
                "Duration": duration,
                "WaitingTime": waiting_time,
                "TimeLoss": time_loss,
                "DepartDelay": depart_delay
            }
            return stats
    return {}    