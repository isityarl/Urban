from sumolib import checkBinary
import traci

sumo_cfg = "src/data/osm.sumocfg"

traci.start([checkBinary("sumo"), "-c", sumo_cfg])

tls_ids = traci.trafficlight.getIDList()

print("Traffic-light intersections:", tls_ids)
print(len(tls_ids))

traci.close()
