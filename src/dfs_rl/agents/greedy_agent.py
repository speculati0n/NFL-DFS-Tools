import numpy as np

class GreedyAgent:
    """Agent that always selects the highest-projected valid player."""
    def __init__(self, projections: np.ndarray):
        # flatten to 1D numpy array for indexing
        self.proj = np.asarray(projections, dtype=float)

    def act(self, mask):
        idx = np.where(mask == 1)[0]
        if len(idx) == 0:
            return 0
        best = idx[np.argmax(self.proj[idx])]
        return int(best)
