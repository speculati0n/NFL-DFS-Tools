import os
from typing import Optional, Dict, Any

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
from stack_metrics import analyze_lineup, compute_presence_and_counts, compute_features, classify_bucket
from dfs.rl_reward import compute_reward, compute_partial_reward

# explicit slot order
SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

class DKNFLEnv(gym.Env if gym else object):
    """Environment for sequential DraftKings NFL lineup construction."""

    metadata = {"render_modes": []}

    def __init__(self, player_pool: pd.DataFrame, min_salary_pct: float = None, rl_reward_cfg: Dict[str,Any] | None = None):
        self.min_salary_pct = (
            float(min_salary_pct)
            if min_salary_pct is not None
            else float(os.getenv("MIN_SALARY_PCT", DEFAULT_MIN_SPEND_PCT))
        )
        self.pool = player_pool.reset_index(drop=True).copy()
        self.pool["salary"] = self.pool["salary"].apply(sanitize_salary)
        self.pool["projections_proj"] = self.pool["projections_proj"].astype(float)

        self.rl_reward_cfg = (rl_reward_cfg or {}).get("weights", rl_reward_cfg or {})

        self.players = []
        self.player_actpts: list[float] = []
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
            self.player_actpts.append(float(row.get("projections_actpts", 0.0)))

        self.n = len(self.players)
        if gym:
            self.action_space = spaces.Discrete(self.n)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )

        self.reset()

    # --- helpers -------------------------------------------------
    def _empty_row_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "salary": 0,
            "projections_proj": 0.0,
            "projections_actpts": 0.0,
        }

        for slot in SLOTS:
            row[slot] = None
            row[f"{slot}_team"] = None
            row[f"{slot}_opp"] = None
            row[f"{slot}_pos"] = None
        return row

    def _apply_action(self, action: int):
        slot = SLOTS[self.slot_idx]
        p = self.players[action]
        setattr(self.lineup, slot, p)
        self.cur_row[slot] = p.name
        self.cur_row[f"{slot}_team"] = p.team
        self.cur_row[f"{slot}_opp"] = p.opp
        self.cur_row[f"{slot}_pos"] = p.pos
        self.cur_row["salary"] += p.salary
        self.cur_row["projections_proj"] += p.proj
        self.cur_row["projections_actpts"] += self.player_actpts[action]

        self.slot_idx += 1
        return slot, p

    def _push_preview(self, action: int):
        slot, p = self._apply_action(action)
        self._preview_stack.append((slot, p))

    def _pop_preview(self):
        if not self._preview_stack:
            return
        slot, p = self._preview_stack.pop()
        self.slot_idx -= 1
        setattr(self.lineup, slot, None)
        self.cur_row[slot] = None
        self.cur_row[f"{slot}_team"] = None
        self.cur_row[f"{slot}_opp"] = None
        self.cur_row[f"{slot}_pos"] = None
        self.cur_row["salary"] -= p.salary
        self.cur_row["projections_proj"] -= p.proj
        # use player's id to index act pts list
        self.cur_row["projections_actpts"] -= self.player_actpts[int(p.id)]


    def _lineup_complete(self) -> bool:
        return self.slot_idx >= len(SLOTS)

    def _obs(self):
        return np.array([0.0], dtype=np.float32), {"action_mask": self._mask()}

    def _mask(self) -> np.ndarray:
        if self.slot_idx >= len(SLOTS):
            return np.zeros(self.n, dtype=np.int8)
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
        self.cur_row = self._empty_row_dict()
        self._preview_stack: list[tuple[str, Player]] = []
        self._last_partial_reward = 0.0
        return self._obs()

    def step(self, action: int):
        mask = self._mask()
        if action < 0 or action >= self.n or mask[action] == 0:
            obs, info = self._obs()
            info["partial_reward"] = self._last_partial_reward
            return obs, -0.01, False, False, info

        self.picks.append(action)
        self._apply_action(action)

        prev = self._last_partial_reward
        new_r = compute_partial_reward(self.cur_row, self.rl_reward_cfg)
        delta = new_r - prev
        self._last_partial_reward = new_r

        done = self._lineup_complete()
        reward = float(delta)
        info_extra: Dict[str, Any] = {}
        if done:
            full_r = compute_reward(self.cur_row, self.rl_reward_cfg)
            reward += float(full_r - new_r)
            if not validate_lineup(
                self.lineup, cap=DEFAULT_SALARY_CAP, min_pct=self.min_salary_pct
            ):
                reward += -100.0
            flags, _ = compute_presence_and_counts(self.cur_row)
            feats = compute_features(self.cur_row)
            bucket = classify_bucket(flags)
            self.cur_row["stack_bucket"] = bucket
            self.cur_row["double_te"] = feats.get("feat_double_te", 0)
            self.cur_row["flex_pos"] = feats.get("flex_pos", "")
            self.cur_row["dst_conflicts"] = feats.get("feat_any_vs_dst", 0)

            info_extra.update(self.cur_row)
            info_extra["lineup_indices"] = self.picks.copy()

        obs, info = self._obs()
        info.update(info_extra)
        info["partial_reward"] = self._last_partial_reward
        return obs, reward, done, False, info
