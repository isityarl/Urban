import subprocess
from back.app.config import SUMO_BINARY, SUMO_CONFIG

SUMO_BIN = "/home/yarl/Desktop/sumo/bin/sumo"

def run_sumo():
    cmd = [SUMO_BINARY, "-c", SUMO_CONFIG]
    subprocess.run([
        SUMO_BIN, "-c", SUMO_CONFIG, "--start", "--no-step-log", "--no-warnings", "--duration-log.statistics", "--verbose"
    ], check=True)