import random
import xml.etree.ElementTree as ET
import sumolib
from pathlib import Path

NET_FILE = "data/almaty.net.xml"
ALL_EDGES_FILE = "data/all_edges.txt"
MAIN_EDGES_FILE = "data/main_edges.txt"
BORDER_FILE = "data/border_edges.txt"
OUTPUT_FILE = "data/routes_custom.rou.xml"

BEGIN = 0
END = 3600

N_COVERAGE = 3000          # anywhere <-> anywhere
N_BORDER_TO_ALL = 4000     # entering -> anywhere
N_ALL_TO_BORDER = 4000     # anywhere -> leaving
N_BORDER_TO_BORDER = 10000  # entering -> leaving

def indent(elem, level=0):
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def read_edge_list(path):
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

def read_border_edges(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f]
    if "" in lines:
        split_idx = lines.index("")
        leaving = [e for e in lines[:split_idx] if e]
        entering = [e for e in lines[split_idx+1:] if e]
    else:
        raise ValueError("Need a blank line separating leaving and entering edges")
    return entering, leaving

def add_vehicle(net, root, vid, from_id, to_id, begin, end):
    try:
        from_edge = net.getEdge(from_id)
        to_edge = net.getEdge(to_id)
    except KeyError:
        return False

    path = net.getShortestPath(from_edge, to_edge)[0]
    if not path:
        return False

    edges_str = " ".join(e.getID() for e in path)
    depart_time = random.uniform(begin, end)

    veh = ET.SubElement(
        root,
        "vehicle",
        {
            "id": vid,
            "depart": f"{depart_time:.2f}",
            "departLane": "best",
            "departSpeed": "max",
        },
    )
    ET.SubElement(veh, "route", {"edges": edges_str})
    return True

def main():
    net = sumolib.net.readNet(NET_FILE)

    all_edges = read_edge_list(ALL_EDGES_FILE)
    main_edges = read_edge_list(MAIN_EDGES_FILE)  # not yet used, but ready
    entering_edges, leaving_edges = read_border_edges(BORDER_FILE)

    root = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )

    vehicles_created = 0

    # 1) coverage: anywhere -> anywhere (from all_edges)
    max_tries = 50
    for i in range(N_COVERAGE):
        for _ in range(max_tries):
            f = random.choice(all_edges)
            t = random.choice(all_edges)
            if f == t:
                continue
            vid = f"cov_{vehicles_created}"
            if add_vehicle(net, root, vid, f, t, BEGIN, END):
                vehicles_created += 1
                break

    # 2) border -> anywhere (entering -> all_edges)
    for i in range(N_BORDER_TO_ALL):
        for _ in range(max_tries):
            f = random.choice(entering_edges)
            t = random.choice(all_edges)
            if f == t:
                continue
            vid = f"b2a_{vehicles_created}"
            if add_vehicle(net, root, vid, f, t, BEGIN, END):
                vehicles_created += 1
                break

    # 3) anywhere -> border (all_edges -> leaving)
    for i in range(N_ALL_TO_BORDER):
        for _ in range(max_tries):
            f = random.choice(all_edges)
            t = random.choice(leaving_edges)
            if f == t:
                continue
            vid = f"a2b_{vehicles_created}"
            if add_vehicle(net, root, vid, f, t, BEGIN, END):
                vehicles_created += 1
                break

    # 4) border -> border (entering -> leaving)
    for i in range(N_BORDER_TO_BORDER):
        for _ in range(max_tries):
            f = random.choice(entering_edges)
            t = random.choice(leaving_edges)
            if f == t:
                continue
            vid = f"b2b_{vehicles_created}"
            if add_vehicle(net, root, vid, f, t, BEGIN, END):
                vehicles_created += 1
                break

    indent(root)
    tree = ET.ElementTree(root)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    tree.write(OUTPUT_FILE, encoding="UTF-8", xml_declaration=True)
    print(f"Created {vehicles_created} vehicles in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
