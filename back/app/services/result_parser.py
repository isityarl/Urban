import xml.etree.ElementTree as ET
from back.app.config import TRIPINFO_FILE

def parse_tripinfo():
    tree = ET.parse(TRIPINFO_FILE)
    root = tree.getroot()
    
    trip = root.find("tripinfo")
    if trip is None:
        return {"error": "no tripinfo"}
    
    return {
        "duration_sec": float(trip.attrib.get("duration", 0)),
        "route_length_m": float(trip.attrib.get("routeLength", 0)),
        "waiting_time_sec": float(trip.attrib.get("waitingTime", 0)),
    }