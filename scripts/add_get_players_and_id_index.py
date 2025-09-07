#!/usr/bin/env python3
# scripts/add_get_players_and_id_index.py
#
# Idempotent patch for src/nfl_optimizer.py:
#  - Add Player dataclass if missing
#  - Build self.id_to_key after player_dict creation
#  - Implement NFL_Optimizer.get_players(ids)
#  - Ensure salary guard (already inserted previously) works off the slotted nine

from pathlib import Path
import re
import sys

OPT = Path("src/nfl_optimizer.py")

PLAYER_CLASS = r"""
# --- begin Player dataclass (optimizer) ---
try:
    from dataclasses import dataclass
except Exception:
    dataclass = None

if dataclass is not None and "class Player(" not in globals():
    @dataclass
    class Player:
        name: str
        pos: str
        team: str
        salary: float
        proj: float
        ceil: float = 0.0
        stddev: float = 0.0
        own: float = 0.0
        id: int = 0
        key: str = ""
# --- end Player dataclass (optimizer) ---
"""

GET_PLAYERS_DEF = r"""
    # === begin get_players helper ===
    def get_players(self, ids):
        """
        Map LP variable IDs -> Player objects using player_dict + id_to_key.
        """
        out = []
        # Build index lazily if needed
        if not hasattr(self, "id_to_key") or not self.id_to_key:
            try:
                self.id_to_key = { rec["ID"]: key for key, rec in self.player_dict.items() }
            except Exception:
                self.id_to_key = {}
        for _pid in ids:
            try:
                pid = int(str(_pid).strip())
            except Exception:
                continue
            key = self.id_to_key.get(pid)
            if key is None:
                # fallback: scan (slower, but safe)
                for k, rec in self.player_dict.items():
                    try:
                        if int(rec.get("ID", -1)) == pid:
                            key = k
                            break
                    except Exception:
                        pass
            if key is None:
                continue
            (name, pos_str, team) = key
            rec = self.player_dict.get(key, {})
            # safe fetches
            def _sf(x, d=0.0):
                try:
                    if x is None:
                        return float(d)
                    return float(x)
                except Exception:
                    return float(d)
            def _si(x, d=0):
                try:
                    return int(float(x))
                except Exception:
                    return int(d)

            pos = str(rec.get("Position", pos_str)).upper()
            team_abbrev = str(rec.get("TeamAbbrev", team)).upper()
            p = Player(
                name=str(name),
                pos=pos,
                team=team_abbrev,
                salary=_sf(rec.get("Salary", 0.0), 0.0),
                proj=_sf(rec.get("Fpts", 0.0), 0.0),
                ceil=_sf(rec.get("Ceil", rec.get("CEIL", 0.0)), 0.0),
                stddev=_sf(rec.get("StdDev", rec.get("STDDEV", 0.0)), 0.0),
                own=_sf(rec.get("Own", rec.get("OWN", 0.0)), 0.0),
                id=_si(rec.get("ID", pid), pid),
                key=str(rec.get("ID", pid)),
            )
            out.append(p)
        return out
    # === end get_players helper ===
"""

def ensure_player_class(src: str) -> str:
    if re.search(r"\bclass\s+Player\b", src):
        return src
    # insert after imports
    m = re.search(r"(\nimport[^\n]*\n(?:from[^\n]*\n|import[^\n]*\n)*)", src)
    insert_at = m.end() if m else 0
    return src[:insert_at] + PLAYER_CLASS + src[insert_at:]

def add_id_index_after_player_dict_build(src: str) -> str:
    # Find where player_dict entries are created and inject id_to_key build once finished.
    # Anchor on a line that looks like: self.player_dict[(row['name'], pos_str, team)] = { ... }
    pat_block_end = re.compile(r"\n\s*#\s*End\s+player\s+pool.*\n", re.IGNORECASE)
    m_end = pat_block_end.search(src)
    if m_end and "self.id_to_key" not in src:
        inject = (
            "\n        # Fast lookup: var ID -> (name, pos, team)\n"
            "        self.id_to_key = { rec['ID']: key for key, rec in self.player_dict.items() }\n"
        )
        return src[:m_end.end()] + inject + src[m_end.end():]
    # Fallback: inject right after 'self.problem += Salary...' constraints section (already exists)
    m_any = re.search(r"\n\s*self\.problem\s*\+=\s*.*MinSalary.*\n", src)
    if m_any and "self.id_to_key" not in src:
        inject = (
            "\n        # Build id->key index after pool setup\n"
            "        self.id_to_key = { rec['ID']: key for key, rec in self.player_dict.items() }\n"
        )
        return src[:m_any.end()] + inject + src[m_any.end():]
    return src

def ensure_get_players_method(src: str) -> str:
    if re.search(r"\ndef\s+get_players\s*\(", src):
        return src
    # Inject inside NFL_Optimizer class: after __init__ or right before select_slot_players
    m_cls = re.search(r"\nclass\s+NFL_Optimizer\([^\)]*\):\s*\n", src)
    if not m_cls:
        return src
    insert_pos = m_cls.end()
    # If __init__ exists, place method after it
    m_init = re.search(r"\n\s*def\s+__init__\s*\(self[^\)]*\):", src[m_cls.end():])
    if m_init:
        insert_pos = m_cls.end() + m_init.end()
        # move to end of __init__ block
        # naive: find next 'def ' at same indent
        tail = src[insert_pos:]
        m_next = re.search(r"\n\s*def\s+\w+\s*\(", tail)
        if m_next:
            insert_pos = m_cls.end() + m_init.end() + m_next.start()
    return src[:insert_pos] + GET_PLAYERS_DEF + src[insert_pos:]

def main():
    if not OPT.exists():
        print("Could not find src/nfl_optimizer.py. Run from repo root.", file=sys.stderr)
        sys.exit(2)

    src = OPT.read_text(encoding="utf-8")
    orig = src

    src = ensure_player_class(src)
    src = add_id_index_after_player_dict_build(src)
    src = ensure_get_players_method(src)

    if src != orig:
        bak = OPT.with_suffix(".py.bak")
        bak.write_text(orig, encoding="utf-8")
        OPT.write_text(src, encoding="utf-8")
        print(f"✅ Patched {OPT} (backup at {bak})")
    else:
        print("ℹ️ No changes applied (already present).")

if __name__ == "__main__":
    main()
