import json
from math import radians, cos, sin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * sqrt(a)

class EdgeMatcher:
    def __init__(self, geojson_path):
        with open(geojson_path) as f:
            self.edges = json.load(f)["features"]

    def nearest_edge(self, lat, lon):
        best_edge = None
        best_dist = float("inf")

        for feat in self.edges:
            coords = feat["geometry"]["coordinates"]

            for x, y in coords:
                d = haversine(lat, lon, y, x)
                if d < best_dist:
                    best_dist = d
                    best_edge = feat["properties"]["edge_id"]

        return best_edge
