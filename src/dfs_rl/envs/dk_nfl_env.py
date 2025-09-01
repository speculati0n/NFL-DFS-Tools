import os
from typing import Optional

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - gym is optional
    gym = None
    spaces = object

from dfs.constraints import (
    Player,
    Lineup,
    action_mask_for_slot,
    validate_lineup,
    sanitize_salary,
    DEFAULT_SALARY_CAP,
    DEFAULT_MIN_SPEND_PCT,
)

from dfs_rl.utils.lineup import SLOTS

class DKNFLEnv(gym.Env if gym else object):
    """Environment for sequential DraftKings NFL lineup construction."""

    metadata = {"render_modes": []}

    def __init__(self, player_pool: pd.DataFrame, min_salary_pct: float = None):
        self.min_salary_pct = (
            float(min_salary_pct)
            if min_salary_pct is not None
            else float(os.getenv("MIN_SALARY_PCT", DEFAULT_MIN_SPEND_PCT))
        )
        self.pool = player_pool.reset_index(drop=True).copy()
        self.pool["salary"] = self.pool["salary"].apply(sanitize_salary)
        self.pool["projections_proj"] = self.pool["projections_proj"].astype(float)

        self.players = []
        self.pool_by_pos = {"QB": [], "RB": [], "WR": [], "TE": [], "DST": []}
        for idx, row in self.pool.iterrows():
            p = Player(
                id=str(idx),
                name=row["name"],
                pos=row["pos"],
                team=row.get("team"),
                opp=row.get("opp"),
                salary=int(row["salary"]),
                proj=float(row["projections_proj"]),
            )
            self.players.append(p)
            self.pool_by_pos[p.pos].append(p)

        self.n = len(self.players)
        if gym:
            self.action_space = spaces.Discrete(self.n)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )

        self.reset()

    def _mask(self) -> np.ndarray:
        slot = SLOTS[self.slot_idx]
        used_ids = {p.id for p in self.lineup.players()}
        mask_dict = action_mask_for_slot(
            slot,
            self.lineup,
            self.pool_by_pos,
            used_ids,
            cap=DEFAULT_SALARY_CAP,
            min_pct=self.min_salary_pct,
        )
        mask = np.zeros(self.n, dtype=np.int8)
        for i, p in enumerate(self.players):
            if mask_dict.get(p.id, False):
                mask[i] = 1
        return mask

    def reset(self, *, seed: Optional[int] = None, options=None):  # pragma: no cover - gym API
        self.lineup = Lineup()
        self.slot_idx = 0
        self.picks: list[int] = []
        return np.array([0.0], dtype=np.float32), {"action_mask": self._mask()}

    def step(self, action: int):
        mask = self._mask()
        if action < 0 or action >= self.n or mask[action] == 0:
            return (
                np.array([0.0], dtype=np.float32),
                -0.01,
                False,
                False,
                {"action_mask": mask},
            )

        p = self.players[action]
        slot = SLOTS[self.slot_idx]
        setattr(self.lineup, slot, p)
        self.picks.append(action)
        self.slot_idx += 1
        done = self.slot_idx >= len(SLOTS)

        if done:
            sal = self.lineup.salary()
            if not validate_lineup(
                self.lineup,
                cap=DEFAULT_SALARY_CAP,
                min_pct=self.min_salary_pct,
            ):
                reward = -100.0
            else:
                reward = self.lineup.projection() - (
                    (DEFAULT_SALARY_CAP - sal) / 1000.0
                )
            return (
                np.array([1.0], dtype=np.float32),
                reward,
                True,
                False,
                {"lineup_indices": self.picks, "lineup_salary": sal},
            )
        else:
            return (
                np.array([0.0], dtype=np.float32),
                0.0,
                False,
                False,
                {"action_mask": self._mask()},
            )
