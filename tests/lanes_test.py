from src.RL_env.env import SumoEnv
import traci

TLS_TO_TEST = [
    "joinedS_cluster_10111825049_1363427911_1363427914_254854089_cluster_1363427912_1363427913_254854087",
    "joinedS_cluster_11204795714_11204795717_254027086_5137419282_cluster_11204795715_11625890456_254027085_5137419281",
]

if __name__ == "__main__":
    cfg_path = "src/data/osm.sumocfg"
    net_path = "src/data/osm.net.xml.gz"

    env = SumoEnv(cfg_path=cfg_path, net_path=net_path, gui=True, step_length=1)
    state = env.reset()

    traci.simulationStep()
    state = env.get_state()

    for tls in TLS_TO_TEST:
        if tls not in env.main_tls:
            print(f"{tls} is not in env.main_tls")
            continue

        lanes = env.lanes_by_tls.get(tls, [])
        print(f"\nTLS: {tls}")
        print("  ordered lanes:", lanes[:4])

        if tls in state:
            tls_state = state[tls]
            q = tls_state[0:4]
            wait = tls_state[4:8]
            print("  q (first 4):   ", q)
            print("  wait (first 4):", wait)

    env.close()
