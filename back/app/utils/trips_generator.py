def generate_trip_xml(from_edge: str, to_edge: str):
    return f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <routes>
    <vType id=\"veh_passenger\" vClass=\"passenger\"/>
    <trip id=\"veh0\" type=\"veh_passenger\"
    depart=\"0\"
    departLane=\"best\"
    from=\"{from_edge}\"
    to=\"{to_edge}\"/>
    </routes>
    """