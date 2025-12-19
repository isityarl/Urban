import xml.etree.ElementTree as ET

def generate_trip_xml(from_edge: str, to_edge: str):
    return f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <routes>
    <vType id=\"veh_passenger\" vClass=\"passenger\"/>
    <trip id=\"specialveh\" type=\"veh_passenger\"
    depart=\"0\"
    departLane=\"best\"
    from=\"{from_edge}\"
    to=\"{to_edge}\"/>
    </routes>
    """

def append_trip_to_file(from_edge, to_edge, filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    for trip in root.findall("trip"):
        if trip.get("id") == "specialveh":
            root.remove(trip)

    ET.SubElement(
        root,
        "trip",
        {
            "id": "specialveh",
            "type": "veh_passenger",
            "depart": "0",
            "departLane": "best",
            "from": from_edge,
            "to": to_edge,
        }
    )

    tree.write(filename, encoding="UTF-8", xml_declaration=True)