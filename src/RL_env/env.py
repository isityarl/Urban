import os
import sys
import numpy as np
import traci
import traci.constants as tc
import xml.etree.ElementTree as ET
import sumolib
import gzip
import time

class SumoEnv:
    MAIN_TLS = [
        'joinedS_cluster_256225070_7372759114_7372759146_7372759149_cluster_256225071_4838579591_7372759144_7372759152',
        'joinedS_cluster_11204795714_11204795717_254027086_5137419282_cluster_11204795715_11625890456_254027085_5137419281',
        'joinedS_cluster_11733423188_12469169192_254241547_cluster_254241552_7937339596_9612540405',
        'joinedS_cluster_10090280950_254241529_4895203645_cluster_11225403220_254241533_4895203634',
        'joinedS_cluster_10111825049_1363427911_1363427914_254854089_cluster_1363427912_1363427913_254854087',
        'cluster_12112602457_254241534_6476883898',
        'cluster_11151745702_12647001094_12647001097_12647001202_#1more',
        'cluster_12469169303_12469169304_12469169310_259988455',
        'cluster_12469169302_12469169308_12469169309_259988456',
        'cluster_12469169140_12469169325_260382566_8143107190',
        'cluster_12469169141_12469169321_260730708_8143107188',
        'cluster_11225403169_12469169162_260730706_8143107194',
        'cluster_12469169163_12469169307_260730707_8143107192',
        'cluster_11107231387_11107231388_254241553_6476883897_#1more',
        'cluster_11204795550_11204795551_11204795552_11204795553_#4more',
        'cluster_11383565809_12469169282_12469169312_253950320',
        'cluster_10090280939_11383565810_254241544_9801684682',
        'cluster_11383565816_12553582494_12553582498_260710611_#1more'
    ]

    def __init__(self, cfg_path, net_path, gui=False, step_length=1):
        self.cfg_path = cfg_path
        self.net_path = net_path
        self.gui = gui
        self.step_length = step_length
        self.max_steps = 1000

        self.controlled_tls = self.find_all_intersections()
        self.main_tls = [tl for tl in self.MAIN_TLS if tl in self.controlled_tls]
        self.small_tls = [tl for tl in self.controlled_tls if tl not in self.main_tls]

        self.phases = self.load_tls_phases(self.net_path)
        self.main_phases = {tl: self.phases[tl] for tl in self.main_tls if tl in self.phases}

        self.rl_phase_map = {}
        for tl, phases in self.phases.items():
            rl_indices = []
            for idx, state in enumerate(phases):
                if any(c in ("g", "G") for c in state):
                    rl_indices.append(idx)
            if not rl_indices:
                rl_indices = list(range(len(phases)))
            self.rl_phase_map[tl] = rl_indices

        self.pending_transition = {}
        self.transition_duration = 2

        self.current = 0

    def find_all_intersections(self):
        net = sumolib.net.readNet(self.net_path)
        tls = [n.getID() for n in net.getTrafficLights()]
        return tls

    def load_tls_phases(self, add_path):
        phases = {}
        try:
            if add_path.endswith(".gz"):
                f = gzip.open(add_path, "rt", encoding="utf-8")
            else:
                f = open(add_path, "rt", encoding="utf-8")
            tree = ET.parse(f)
            root = tree.getroot()
            for tl in root.findall("tlLogic"):
                tls_id = tl.get("id")
                phase_states = [p.get("state") for p in tl.findall("phase")]
                phases[tls_id] = phase_states
            f.close()
        except Exception as e:
            print("Failed to parse net phases:", e)
        return phases

    def reset(self):
        if traci.isLoaded():
            try:
                traci.close()
            except Exception:
                pass

        binary = "sumo-gui" if self.gui else "sumo"
        self.current = 0
        traci.start([binary, "-c", self.cfg_path, "--step-length", str(self.step_length),
                     "--no-step-log", "--start", "--no-warnings", "--scale", "1.5"])
        self.current = 0

        state = self.get_state()
        missing_tls = [tl for tl in self.main_tls if tl not in state]
        steps = 0
        while missing_tls and steps < 50:
            traci.simulationStep()
            state = self.get_state()
            missing_tls = [tl for tl in self.main_tls if tl not in state]
            steps += 1

        if missing_tls:
            raise RuntimeError(f"Some main TLS missing after 50 steps: {missing_tls}")

        return state

    def get_state(self):
        state = {}
        for tls in self.main_tls:
            try:
                lanes = list(traci.trafficlight.getControlledLanes(tls))
            except Exception:
                lanes = []
            lanes = lanes[:4] + [None] * max(0, 4 - len(lanes))

            q = []
            wait = []
            speed = []

            for lane in lanes:
                if lane is None:
                    q.append(0)
                    wait.append(0)
                    speed.append(1.0)
                    continue
                try:
                    q.append(traci.lane.getLastStepHaltingNumber(lane))
                except Exception:
                    q.append(0)
                try:
                    wait.append(traci.lane.getWaitingTime(lane))
                except Exception:
                    wait.append(0)
                try:
                    ms = traci.lane.getLastStepMeanSpeed(lane)
                    mx = traci.lane.getMaxSpeed(lane)
                    speed.append(ms / mx if mx > 0 else 1.0)
                except Exception:
                    speed.append(1.0)

            try:
                phase = traci.trafficlight.getPhase(tls)
            except Exception:
                phase = 0

            try:
                next_switch = traci.trafficlight.getNextSwitch(tls)
                cur_time = traci.simulation.getTime()
                
                phase_duration = traci.trafficlight.getPhaseDuration(tls)
                time_in_phase = max(0.0, phase_duration - max(0.0, next_switch - cur_time))
            except Exception:
                time_in_phase = 0.0

            state[tls] = q + wait + speed + [phase, time_in_phase]

        return state

    def find_transition_phase(self, tl, current_phase_idx, target_phase_idx):
        phases = self.phases.get(tl, [])
        if not phases or current_phase_idx is None:
            return None

        cur_state = phases[current_phase_idx]
        cur_green_pos = {i for i, c in enumerate(cur_state) if c in ("g", "G")}

        for idx, cand_state in enumerate(phases):
            if idx == current_phase_idx or idx == target_phase_idx:
                continue
            ok = True
            for pos in cur_green_pos:
                if pos >= len(cand_state) or cand_state[pos] not in ("y", "Y"):
                    ok = False
                    break
            if ok:
                return idx
        return None

    def request_phase_change(self, tl, rl_action_idx):
        
        if tl not in self.rl_phase_map:
            return

        rl_map = self.rl_phase_map[tl]
        if rl_action_idx < 0 or rl_action_idx >= len(rl_map):
            return

        target_phase_idx = rl_map[rl_action_idx]
        try:
            current_phase = traci.trafficlight.getPhase(tl)
        except Exception:
            current_phase = None

        if current_phase == target_phase_idx:
            if tl in self.pending_transition:
                del self.pending_transition[tl]
            return

        transition_idx = self.find_transition_phase(tl, current_phase, target_phase_idx)
        if transition_idx is None:
            self.pending_transition[tl] = {"target": target_phase_idx, "transition": None, "steps_left": 1}
        else:
            self.pending_transition[tl] = {"target": target_phase_idx, "transition": transition_idx, "steps_left": self.transition_duration}

    def apply_pending_transitions_now(self):
        for tls, info in list(self.pending_transition.items()):
            if info["transition"] is not None and info["steps_left"] == self.transition_duration:
                try:
                    traci.trafficlight.setPhase(tls, info["transition"])
                except Exception:
                    pass
                info["steps_left"] -= 1
            elif info["transition"] is not None and info["steps_left"] > 0:
                try:
                    traci.trafficlight.setPhase(tls, info["transition"])
                except Exception:
                    pass
                info["steps_left"] -= 1
            elif info["transition"] is not None and info["steps_left"] <= 0:
                try:
                    traci.trafficlight.setPhase(tls, info["target"])
                except Exception:
                    pass
                del self.pending_transition[tls]
            elif info["transition"] is None:
                try:
                    traci.trafficlight.setPhase(tls, info["target"])
                except Exception:
                    pass
                del self.pending_transition[tls]

    def apply_small_tls_heuristic(self):
        for tls in self.small_tls:
            if tls not in self.phases:
                continue

            phases = self.phases[tls]
            links = traci.trafficlight.getControlledLinks(tls)

            if not phases or not links:
                continue

            best_phase = 0
            best_queue = -1

            for p_idx, phase_state in enumerate(phases):
                green_lanes = set()

                for sig_idx, signal_char in enumerate(phase_state):
                    if signal_char not in ("g", "G"):
                        continue
                    if sig_idx >= len(links):
                        continue
                    if not links[sig_idx]:
                        continue

                    in_lane = links[sig_idx][0][0]
                    green_lanes.add(in_lane)

                if not green_lanes:
                    continue

                q = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in green_lanes)

                if q > best_queue:
                    best_queue = q
                    best_phase = p_idx
            try:
                traci.trafficlight.setPhase(tls, best_phase)
            except Exception:
                pass

    def get_reward(self, state=None):
        if state is None:
            state = self.get_state()

        rewards = {}

        alpha = 1.0
        beta = 0.25
        gamma = 0.1
        delta = 0.8
        epsilon = 0.05

        for tls in self.main_tls:
            tls_state = state[tls]
            q = np.array(tls_state[0:4])
            wait = np.array(tls_state[4:8])
            speed = np.array(tls_state[8:12])
            phase = tls_state[12]
            time_in_phase = tls_state[13]

            total_queue = np.sum(q)
            total_wait = np.sum(wait)
            avg_speed = np.mean(speed)

            lane_imbalance = np.std(q)

            phase_penalty = max(time_in_phase - 10, 0)

            reward = -(alpha * total_queue +
                       beta * total_wait +
                       delta * lane_imbalance +
                       epsilon * phase_penalty) + gamma * avg_speed

            rewards[tls] = float(reward)

        return rewards

    def get_done(self):
        if self.current >= self.max_steps:
            return True
        return False

    def step(self, actions):
        for tls, rl_action in actions.items():
            if tls in self.rl_phase_map:
                try:
                    self.request_phase_change(tls, int(rl_action))
                except Exception:
                    pass
            else:
                try:
                    traci.trafficlight.setPhase(tls, int(rl_action))
                except Exception:
                    pass

        self.apply_pending_transitions_now()
        self.apply_small_tls_heuristic()

        traci.simulationStep()
        self.current += 1

        state = self.get_state()
        rewards = self.get_reward(state)
        done = self.get_done()

        return state, rewards, done, {}

    def close(self):
        try:
            traci.close()
        except Exception:
            pass


if __name__ == "__main__":
    cfg_path = "src/data/osm.sumocfg"
    net_path = "src/data/osm.net.xml.gz"
    env = SumoEnv(cfg_path=cfg_path, net_path=net_path, gui=True, step_length=1)
    state = env.reset()
    for step in range(100):
        actions = {tl: 0 for tl in env.main_tls}
        state, rewards, done, _ = env.step(actions)
        if step % 10 == 0:
            print("Step", step)
            for tls in env.small_tls[:5]:
                try:
                    print(
                        tls,
                        "phase:", traci.trafficlight.getPhase(tls),
                        "state:", traci.trafficlight.getRedYellowGreenState(tls)
                    )
                except Exception:
                    pass
        if done:
            break
    env.close()
