# import multiprocessing as mp
# from src.RL_env.env import SumoEnv

# def worker(remote, cfg, net, gui, step_length):
#     env = SumoEnv(cfg_path=cfg, net_path=net, gui=gui, step_length=step_length)
#     while True:
#         cmd, data = remote.recv()
#         if cmd == "reset":
#             state = env.reset()
#             remote.send({
#                 "state": state,
#                 "main_phases": env.main_phases
#             })
#         elif cmd == "step":
#             remote.send(env.step(data))
#         elif cmd == "close":
#             env.close()
#             remote.close()
#             break

# class ParallelEnvs:
#     def __init__(self, num_envs, cfg, net, gui=False, step_length=1):
#         self.num_envs = num_envs
#         self.remotes, self.workers = zip(*[mp.Pipe() for _ in range(num_envs)])
#         self.ps = []

#         for remote in self.workers:
#             p = mp.Process(target=worker, args=(remote, cfg, net, gui, step_length))
#             p.daemon = True
#             p.start()
#             self.ps.append(p)

#     def reset(self):
#         for r in self.remotes:
#             r.send(("reset", None))
#         return [r.recv() for r in self.remotes]

#     def step(self, actions_list):
#         for r, ac in zip(self.remotes, actions_list):
#             r.send(("step", ac))
#         return [r.recv() for r in self.remotes]

#     def close(self):
#         for r in self.remotes:
#             r.send(("close", None))
#         for p in self.ps:
#             p.join()


import multiprocessing as mp
from src.RL_env.env import SumoEnv

def worker(remote, cfg, net, gui, step_length):
    env = SumoEnv(cfg_path=cfg, net_path=net, gui=gui, step_length=step_length)
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            state = env.reset()
            remote.send({
                "state": state,
                "main_phases": env.main_phases
            })
        elif cmd == "step":
            s, r, d, info = env.step(data)
            remote.send({
                "state": s,
                "rewards": r,
                "done": d,
                "info": info
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break

class ParallelEnvs:
    def __init__(self, num_envs, cfg, net, gui=False, step_length=1):
        self.num_envs = num_envs
        self.remotes, self.workers = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = []

        for remote in self.workers:
            p = mp.Process(target=worker, args=(remote, cfg, net, gui, step_length))
            p.daemon = True
            p.start()
            self.ps.append(p)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return [r.recv() for r in self.remotes]

    def step(self, actions_list):
        for r, ac in zip(self.remotes, actions_list):
            r.send(("step", ac))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.ps:
            p.join()
