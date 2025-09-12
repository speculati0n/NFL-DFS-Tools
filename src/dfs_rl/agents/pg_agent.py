import torch
import torch.nn as nn
import torch.optim as optim

class TinyPolicy(nn.Module):
    def __init__(self, n_players: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_players, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_players),
        )

    def forward(self, x):
        return self.net(x)

class PGAgent:
    """REINFORCE policy that learns from shaped rewards."""

    def __init__(self, n_players: int, lr: float = 1e-3, seed: int = 42):
        torch.manual_seed(seed)
        self.n = n_players
        self.policy = TinyPolicy(n_players)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self._states = []
        self._actions = []
        self._rewards = []

    def sample(self, mask):
        x = torch.zeros(1, self.n)
        logits = self.policy(x)
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        masked = logits.masked_fill(~mask_t, -1e9)
        dist = torch.distributions.Categorical(logits=masked)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), logp

    # Interface expected by arena._run_agent
    def act(self, obs, info):
        action, logp = self.sample(info.get("action_mask"))
        self._states.append(obs)
        self._actions.append((None, logp))
        return action

    def train_step(self, obs, reward, done, info):
        self._rewards.append(reward)
        if done:
            self.update(self._states, self._actions, self._rewards)
            self._states.clear()
            self._actions.clear()
            self._rewards.clear()

    def update(self, states, actions, rewards):
        returns = []
        g = 0.0
        for r in reversed(rewards):
            g = float(r) + g
            returns.insert(0, g)
        loss = 0.0
        for (_, logp), R in zip(actions, returns):
            loss = loss + (-logp * torch.tensor(R, dtype=torch.float32))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
