import numpy as np
import pandas as pd
from typing import Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = object

from dfs_rl.utils.lineup import DK_CAP, SLOTS

class DKNFLEnv(gym.Env if gym else object):
    """
    Lineup-construction environment with action masking.
    - player_pool must include: name,pos,team,opp,salary,projections_proj
    - Reward: sum of projections (or actuals if later attached) with a penalty
      when the total salary spent is below 49,300
    - Mask drops actions that make the $49,300 floor unreachable
    """
    metadata = {"render_modes": []}

    def __init__(self, player_pool: pd.DataFrame):
        self.pool = player_pool.reset_index(drop=True).copy()
        self.pool["salary"] = self.pool["salary"].astype(int)
        self.pool["projections_proj"] = self.pool["projections_proj"].astype(float)
        self.idx_by_pos = {
            "QB": self.pool.index[self.pool["pos"]=="QB"].to_list(),
            "RB": self.pool.index[self.pool["pos"]=="RB"].to_list(),
            "WR": self.pool.index[self.pool["pos"]=="WR"].to_list(),
            "TE": self.pool.index[self.pool["pos"]=="TE"].to_list(),
            "DST": self.pool.index[self.pool["pos"]=="DST"].to_list(),
        }
        self.reset()
        n = len(self.pool)
        if gym:
            self.action_space = spaces.Discrete(n)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def _needed_pos(self, slot_idx: int) -> str:
        return "FLEX" if SLOTS[slot_idx] == "FLEX" else SLOTS[slot_idx]

    def _max_possible_salary(self, picks: list) -> int:
        used = set(picks)
        total = int(self.pool.loc[picks, "salary"].sum()) if picks else 0
        for slot in range(len(picks), 9):
            need = SLOTS[slot]
            if need == "FLEX":
                candidates = []
                for p in ("RB", "WR", "TE"):
                    candidates += [j for j in self.idx_by_pos[p] if j not in used]
            else:
                candidates = [j for j in self.idx_by_pos[need] if j not in used]
            if not candidates:
                return total
            j = max(candidates, key=lambda x: int(self.pool.loc[x, "salary"]))
            total += int(self.pool.loc[j, "salary"])
            used.add(j)
        return total

    def _mask(self) -> np.ndarray:
        n = len(self.pool)
        mask = np.zeros(n, dtype=np.int8)
        need = self._needed_pos(self.slot_idx)
        allowed = set()
        if need == "FLEX":
            for p in ("RB","WR","TE"):
                allowed.update(self.idx_by_pos[p])
        else:
            allowed.update(self.idx_by_pos[need])
        remain_slots = 9 - (len(self.picks) + 1)
        for i in allowed:
            if i in self.picks:
                continue
            sal = int(self.pool.loc[i, "salary"])
            # ensure we can still afford remaining slots
            if sal > self.cap - 2000 * max(0, remain_slots):
                continue
            if self._max_possible_salary(self.picks + [i]) >= 49300:
                mask[i] = 1
        return mask

    def reset(self, *, seed: Optional[int]=None, options=None):
        self.picks = []
        self.slot_idx = 0
        self.cap = DK_CAP
        return np.array([0.0], dtype=np.float32), {"action_mask": self._mask()}

    def step(self, action: int):
        mask = self._mask()
        if action < 0 or action >= len(mask) or mask[action] == 0:
            return np.array([0.0], dtype=np.float32), -0.01, False, False, {"action_mask": mask}
        self.picks.append(action)
        self.cap -= int(self.pool.loc[action,"salary"])
        self.slot_idx += 1
        done = self.slot_idx >= 9
        if done:
            reward = float(self.pool.loc[self.picks, "projections_proj"].sum())
            total_salary = DK_CAP - self.cap
            if total_salary < 49300:
                reward -= 49300 - total_salary
            return np.array([1.0], dtype=np.float32), reward, True, False, {"lineup_indices": self.picks}
        else:
            return np.array([0.0], dtype=np.float32), 0.0, False, False, {"action_mask": self._mask()}
