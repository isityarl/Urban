import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from src.agents.PPO import ActorCriticHeads, SharedBody

class PPOAgent:
    def __init__(self, state_size, tls_phases, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.body = SharedBody(state_size).to(self.device)
        self.heads = ActorCriticHeads(128, tls_phases).to(self.device)

        self.tls_phases = tls_phases

        self.policies = {}
        self.optimizers = {}
        self.buffers = {}

        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('gae_lambda', 0.95)
        self.clip_eps = config.get('clip_eps', 0.2)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.lr = config.get('learning_rate', 3e-4)
        self.ppo_epochs = config.get('ppo_epochs', 7)
        self.minibatch_size = config.get('ppo_batch_size', 64)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)


        self.optimizer = optim.AdamW(list(self.body.parameters()) + list(self.heads.parameters()), lr=self.lr)
        self.buffers = {tl: [] for tl in tls_phases}

    @torch.no_grad()
    def select_action(self, state, tls):
        actions = {}
        infos = {}
        for tl in tls:
            s_np = np.asarray(state[tl], dtype=np.float32)
            s = torch.from_numpy(s_np).unsqueeze(0).to(self.device)
            h = self.body(s)
            logits, value = self.heads(h, tl)
            dist = Categorical(logits=logits)
            a = dist.sample()

            actions[tl] = int(a.item())
            infos[tl] = {
                'state': s_np,
                'action': int(a.item()),
                'logp': float(dist.log_prob(a).cpu().item()),
                'value': float(value.cpu().item())
            }
        return actions, infos

    def store(self, infos, rewards, done):
        d = float(done)
        for tl, data in infos.items():
            self.buffers[tl].append({
                'state': np.array(data['state'], dtype=np.float32),
                'action': int(data['action']),
                'logp': float(data['logp']),
                'value': float(data['value']),
                'reward': float(rewards[tl] / 50),
                'done': d
            })

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

    def update(self, last_states=None, done=False, verbose=False):
        stats = {}
        for tl, buf in self.buffers.items():
            if len(buf) == 0:
                continue

            states  = np.stack([b['state']  for b in buf])
            actions = np.array([b['action'] for b in buf], dtype=np.int64)
            old_logp = np.array([b['logp']   for b in buf], dtype=np.float32)
            rewards = np.array([b['reward'] for b in buf], dtype=np.float32)
            values  = np.array([b['value']  for b in buf], dtype=np.float32)
            dones   = np.array([b['done']   for b in buf], dtype=np.float32)

            if done:
                last_value = 0.0
            else:
                if last_states is not None and tl in last_states:
                    s_np = np.asarray(last_states[tl], dtype=np.float32)
                    s_t = torch.from_numpy(s_np).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        h_t = self.body(s_t)
                        _, last_value_t = self.heads(h_t, tl)
                        last_value = float(last_value_t.cpu().item())
                else:
                    last_value = float(values[-1]) if len(values) > 0 else 0.0

            adv, rets = self._compute_gae(rewards, values, dones, last_value)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            states_t  = torch.from_numpy(states).to(self.device)
            actions_t = torch.from_numpy(actions).to(self.device)
            old_logp_t = torch.from_numpy(old_logp).to(self.device)
            adv_t     = torch.from_numpy(adv).to(self.device)
            rets_t    = torch.from_numpy(rets).to(self.device)

            N = states_t.size(0)
            idxs = np.arange(N)

            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            count = 0

            params = list(self.body.parameters()) + list(self.heads.parameters())
            for _ in range(self.ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, N, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_idx = idxs[start:end]
                    if len(mb_idx) == 0:
                        continue

                    s_mb = states_t[mb_idx]
                    a_mb = actions_t[mb_idx]
                    old_logp_mb = old_logp_t[mb_idx]
                    adv_mb = adv_t[mb_idx]
                    ret_mb = rets_t[mb_idx]

                    logits, values_pred = self.heads(self.body(s_mb), tl)
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(a_mb)
                    ratio = torch.exp(logp - old_logp_mb)

                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = nn.MSELoss()(values_pred, ret_mb)
                    entropy = dist.entropy().mean()

                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                
                    nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                    self.optimizer.step()

                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
                    epoch_entropy += entropy.item()
                    count += 1

            if count > 0:
                stats[tl] = {
                    'policy_loss': epoch_policy_loss / count,
                    'value_loss': epoch_value_loss / count,
                    'entropy': epoch_entropy / count,
                    'samples': N
                }
        if verbose:
            return stats

    def clear_buffers(self):
        for tl in self.buffers:
            self.buffers[tl] = []
