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
        logits = self.net(x)
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        masked = logits.masked_fill(~mask_t, -1e9)
        probs = torch.softmax(masked, dim=-1)
        a = torch.distributions.Categorical(probs).sample().item()
        self.last = (logits, a, mask_t)
        return a

    def update(self, ret: float):
        logits, a, m = self.last
        masked = logits.masked_fill(~m, -1e9)
        logp = torch.log_softmax(masked, dim=-1)[0, a]
        loss = -logp * torch.tensor(ret, dtype=torch.float32)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
