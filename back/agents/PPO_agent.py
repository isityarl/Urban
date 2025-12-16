import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from back.agents.PPO import ActorCriticHeads, SharedBody

class PPOAgent:
    def __init__(self, state_size, tls_phases, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.body = SharedBody(state_size).to(self.device)
        self.heads = ActorCriticHeads(128, tls_phases).to(self.device)

        self.tls_phases = tls_phases

        self.gamma = config.get("gamma", 0.99)
        self.lam = config.get("gae_lambda", 0.95)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.ent_coef = config.get("ent_coef", 0.01)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.lr = config.get("learning_rate", 3e-4)
        self.ppo_epochs = config.get("ppo_epochs", 10)
        self.minibatch_size = config.get("ppo_batch_size", 64)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.reward_scale = config.get("reward_scale", 10000.0)

        self.optimizer = optim.AdamW(list(self.body.parameters()) + list(self.heads.parameters()), lr=self.lr)
        self.buffers = {tl: [] for tl in tls_phases}

    #action select
    @torch.no_grad()
    def select_action(self, state, tls, deterministic=False):
        actions = {}
        infos = {}
        
        for tl in tls:
            s_np = np.asarray(state[tl], dtype=np.float32)
            s = torch.from_numpy(s_np).unsqueeze(0).to(self.device)

            h = self.body(s)
            logits, value = self.heads(h, tl)
            dist = Categorical(logits=logits)

            if deterministic:
                a = torch.argmax(logits, dim=-1)
                logp = dist.log_prob(a)
            else:
                a = dist.sample()
                logp = dist.log_prob(a)    

            actions[tl] = int(a.item())
            infos[tl] = {
                'state': s_np,
                'action': int(a.item()),
                'logp': float(logp.cpu().item()),
                'value': float(value.cpu().item())
            }
        return actions, infos

    #store transitions
    def store(self, infos, rewards, done, time_limit):
        d = float(done)

        for tl, data in infos.items():
            self.buffers[tl].append({
                "state": data["state"].copy(),
                "action": int(data["action"]),
                "logp": float(data["logp"]),
                "value": float(data["value"]),
                "reward": float(rewards[tl] / self.reward_scale),
                "done": float(done and not time_limit),
            })

    #GAE compute
    def _compute_gae(self, rewards, values, dones, last_value):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t+1] if (t+1) < T else last_value
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.lam * next_nonterminal * last_gae
            adv[t] = last_gae
            
        returns = adv + values
        return adv, returns

    #PPo updte
    def update(self, last_states=None, done=False, verbose=False):
        stats = {}

        params = list(self.body.parameters()) + list(self.heads.parameters())

        for tl, buf in self.buffers.items():
            if len(buf) == 0:
                continue

            states = np.stack([b["state"] for b in buf])
            actions = np.array([b["action"] for b in buf], dtype=np.int64)
            old_logp = np.array([b["logp"] for b in buf], dtype=np.float32)
            rewards = np.array([b["reward"] for b in buf], dtype=np.float32)
            values = np.array([b["value"] for b in buf], dtype=np.float32)
            dones = np.array([b["done"] for b in buf], dtype=np.float32)

            # Bootstrap value
            if done:
                last_value = 0.0
            else:
                if last_states is not None and tl in last_states:
                    s_np = np.asarray(last_states[tl], dtype=np.float32)
                    s = torch.from_numpy(s_np).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        h = self.body(s)
                        _, v = self.heads(h, tl)
                        last_value = float(v.item())
                else:
                    last_value = float(values[-1])

            adv, rets = self._compute_gae(rewards, values, dones, last_value)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # adv = adv

            states_t = torch.from_numpy(states).to(self.device)
            actions_t = torch.from_numpy(actions).to(self.device)
            old_logp_t = torch.from_numpy(old_logp).to(self.device)
            adv_t = torch.from_numpy(adv).to(self.device)
            rets_t = torch.from_numpy(rets).to(self.device)

            N = states_t.size(0)
            idxs = np.arange(N)

            pol_loss_sum = 0.0
            val_loss_sum = 0.0
            ent_sum = 0.0
            count = 0
            grad_norm_sum = 0.0

            for _ in range(self.ppo_epochs):
                np.random.shuffle(idxs)

                for start in range(0, N, self.minibatch_size):
                    self.optimizer.zero_grad()
                    mb_idx = idxs[start:start + self.minibatch_size]
                    if len(mb_idx) == 0:
                        continue

                    s_mb = states_t[mb_idx]
                    a_mb = actions_t[mb_idx]
                    old_logp_mb = old_logp_t[mb_idx]
                    adv_mb = adv_t[mb_idx]
                    ret_mb = rets_t[mb_idx]

                    h = self.body(s_mb)
                    logits, values_pred = self.heads(h, tl)

                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(a_mb)
                    ratio = torch.exp(logp - old_logp_mb.detach())

                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb

                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    values_pred = values_pred.view(-1)
                    ret_mb = ret_mb.view(-1)
                    value_loss = F.mse_loss(values_pred, ret_mb)
                    entropy = dist.entropy().mean()

                    loss = (policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy)

                    loss.backward()

                    total_norm = 0.0
                    for p in params:
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5

                    grad_norm_sum += total_norm

                    nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                    self.optimizer.step()

                    
                    pol_loss_sum += policy_loss.item()
                    val_loss_sum += value_loss.item()
                    ent_sum += entropy.item()
                    count += 1

            if count > 0:
                stats[tl] = {
                    "policy_loss": pol_loss_sum / count,
                    "value_loss": val_loss_sum / count,
                    "entropy": ent_sum / count,
                    "samples": N,
                    "grad_norm": grad_norm_sum / count
                }

        if verbose:
            return stats

    #clear rollout buffer
    def clear_buffers(self):
        for tl in self.buffers:
            self.buffers[tl] = []
