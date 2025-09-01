import numpy as np
from dfs.rl_reward import compute_partial_reward

class GreedyAgent:
    """Greedy agent that selects the action yielding the largest reward delta."""

    def __init__(self, env, epsilon: float = 0.05):
        self.env = env
        self.eps = float(epsilon)

    def act(self, mask):
        legal = np.where(mask == 1)[0]
        if len(legal) == 0:
            return 0
        if np.random.rand() < self.eps:
            return int(np.random.choice(legal))
        baseline = compute_partial_reward(self.env.cur_row, self.env.rl_reward_cfg)
        best_a, best_s = legal[0], -1e9
        for a in legal:
            self.env._push_preview(int(a))
            s = compute_partial_reward(self.env.cur_row, self.env.rl_reward_cfg) - baseline
            self.env._pop_preview()
            if s > best_s:
                best_s, best_a = s, int(a)
        return int(best_a)
