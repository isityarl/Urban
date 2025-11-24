import xml.etree.ElementTree as ET
import gzip

with gzip.open("data/osm.net.xml.gz", "rt", encoding="utf-8") as f:
    tree = ET.parse(f)
root = tree.getroot()

tls_phases = {}

c = 0
for tl in root.findall("tlLogic"):
    tls_id = tl.get("id")
    phases = [p.get("state") for p in tl.findall("phase")]
    tls_phases[tls_id] = phases
    print(tls_id)
    print(phases)
    if c > 3 : break
    c += 1
