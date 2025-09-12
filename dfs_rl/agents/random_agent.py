import numpy as np

class RandomAgent:
    def __init__(self, salaries: np.ndarray, seed: int = 42, tau: float = 10000.0):
        """Random policy weighted by salaries.

        Parameters
        ----------
        salaries : np.ndarray
            Array of player salaries aligned with action indices.
        seed : int
            RNG seed.
        tau : float
            Temperature for softmax weighting; lower -> more bias toward high salary.
        """
        self.rng = np.random.default_rng(seed)
        self.salaries = salaries.astype(float)
        self.tau = float(tau)

    def act(self, obs, info=None):
        mask = info.get("action_mask") if info and isinstance(info, dict) else obs
        idx = np.where(mask == 1)[0]
        if len(idx) == 0:
            return 0
        sal = self.salaries[idx]
        w = np.exp(sal / self.tau)
        w = w / w.sum()
        return int(self.rng.choice(idx, p=w))
