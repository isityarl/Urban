import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedBody(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ActorCriticHeads(nn.Module):
    def __init__(self, shared_dim, tls_phases):
        super().__init__()
        self.pi_heads = nn.ModuleDict()
        self.v_heads  = nn.ModuleDict()
        for tl, phases in tls_phases.items():
            act_dim = len(phases)
            self.pi_heads[tl] = nn.Linear(shared_dim, act_dim)
            self.v_heads[tl]  = nn.Linear(shared_dim, 1)

    def forward(self, x, tl):
        logits = self.pi_heads[tl](x)
        value  = self.v_heads[tl](x).squeeze(-1)
        return logits, value

