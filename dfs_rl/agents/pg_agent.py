import numpy as np
import torch, torch.nn as nn, torch.optim as optim

class TinyPolicy(nn.Module):
    def __init__(self, n_players: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_players, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_players)
        )

    def forward(self, x):
        return self.net(x)

class PGAgent:
    """Tiny REINFORCE policy that learns a distribution over players; mask invalid actions."""
    def __init__(self, n_players: int, lr: float = 1e-3, seed: int = 42, cfg: dict | None = None):
        torch.manual_seed(seed)
        self.n = n_players
        self.net = TinyPolicy(n_players)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.last = None
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg or {}

    def act(self, mask):
        eps = self.cfg.get("rl", {}).get("epsilon", 0.1)
        T = self.cfg.get("rl", {}).get("softmax_temperature", 0.9)
        legal_idx = np.where(mask == 1)[0]
        if len(legal_idx) == 0:
            return 0
        x = torch.zeros(1, self.n)
        logits = self.net(x).detach().numpy().flatten()
        logits[mask == 0] = -1e9
        if self.rng.random() < eps:
            a = int(self.rng.choice(legal_idx))
        else:
            s = logits[legal_idx] / max(T, 1e-6)
            s = s - s.max()
            p = np.exp(s); p /= p.sum()
            a = int(self.rng.choice(legal_idx, p=p))
        self.last = (torch.tensor(logits).unsqueeze(0), a, torch.tensor(mask, dtype=torch.float32).unsqueeze(0))
        return a

    def update(self, ret: float):
        logits, a, m = self.last
        masked = logits + torch.log(m + 1e-8)
        logp = torch.log_softmax(masked, dim=-1)[0, a]
        loss = -logp * torch.tensor(ret, dtype=torch.float32)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
