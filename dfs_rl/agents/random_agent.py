import numpy as np

class RandomAgent:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def act(self, mask):
        idx = np.where(mask == 1)[0]
        if len(idx) == 0: return 0
        return int(self.rng.choice(idx))
