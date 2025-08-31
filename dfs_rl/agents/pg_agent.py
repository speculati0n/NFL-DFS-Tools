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
    def __init__(self, n_players: int, lr: float = 1e-3, seed: int = 42):
        torch.manual_seed(seed)
        self.n = n_players
        self.net = TinyPolicy(n_players)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.last = None

    def act(self, mask):
        x = torch.zeros(1, self.n)
        logits = self.net(x).detach().numpy().flatten()
        logits[mask == 0] = -1e9
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        a = int(np.random.choice(np.arange(self.n), p=probs))
        self.last = (torch.tensor(logits).unsqueeze(0), a, torch.tensor(mask, dtype=torch.float32).unsqueeze(0))
        return a

    def update(self, ret: float):
        logits, a, m = self.last
        masked = logits + torch.log(m + 1e-8)
        logp = torch.log_softmax(masked, dim=-1)[0, a]
        loss = -logp * torch.tensor(ret, dtype=torch.float32)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
