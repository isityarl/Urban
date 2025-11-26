import os
import sys
import numpy as np
import traci
import traci.constants as tc
import xml.etree.ElementTree as ET
import sumolib
import gzip

class SumoEnv:
    def __init__(self, cfg_path, net_path, gui=False, step_length=1):
        self.cfg_path = cfg_path
        self.net_path = net_path
        self.gui = gui
        self.step_length = step_length
        self.max_steps = 3000

        self.controlled_tls = self.find_all_intersections()
        self.phases = self.load_tls_phases(net_path)


        self.current = 0

    def find_all_intersections(self): # Список светофоров
        net = sumolib.net.readNet(self.net_path)
        tls = [n.getID() for n in net.getTrafficLights()]
        return tls

    
    def load_tls_phases(self, add_path):
        with gzip.open("data/osm.net.xml.gz", "rt", encoding="utf-8") as f:
            tree = ET.parse(f)  
        root = tree.getroot()

        tls_phases = {}

        for tl in root.findall("tlLogic"):
            tls_id = tl.get("id")
            phases = [p.get("state") for p in tl.findall("phase")]
            tls_phases[tls_id] = phases

        return tls_phases
        
    
    def reset(self): # Перезапуск среды + начальное состояние
        if traci.isLoaded():
            traci.close()

        binary = "sumo-gui" if self.gui else "sumo"
        traci.start([binary, "-c", self.cfg_path, "--step-length", str(self.step_length), "--no-step-log","--start", "--no-warnings"])

        self.current = 0
        return self.get_state()

    
    def get_state(self): # Берем инфу о состоянии среды
        state = []
        for tls in self.controlled_tls:
            lanes = traci.trafficlight.getControlledLanes(tls)
            count_on_lane = [traci.lane.getLastStepVehicleNumber(lane) for lane in lanes]
            state.extend(count_on_lane)

            current_phase_id = traci.trafficlight.getPhase(tls)
            state.append(current_phase_id)

        return state

    def step(self, actions): # Делаем шаг с заданным action
        for tls, action in actions.items():
            traci.trafficlight.setPhase(tls, action)

        traci.simulationStep()
        self.current += 1
        state = self.get_state()
        reward = self.get_reward()
        done = self.get_done()

        return state, reward, done, {}
    
    def get_reward(self): # Вычисляем reward
        total_queue = 0
        total_delay = 0

        for tls in self.controlled_tls:
            lanes = traci.trafficlight.getControlledLanes(tls)
            for lane in lanes:
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
                speed = traci.lane.getLastStepMeanSpeed(lane)
                max_speed = traci.lane.getMaxSpeed(lane)
                delay = (1 - speed / max_speed) if max_speed > 0 else 0
                total_delay += delay

        alpha = 1
        beta = 0.25

        reward = -(alpha * total_queue + beta * total_delay) #формула
        return reward

    
    def get_done(self):
        if self.current >= self.max_steps: return True
        return False
    
    def close(self):
        traci.close()