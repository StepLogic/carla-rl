"""The one place the train/held-out town partition is decided.

Frozen to a committed JSON rather than regenerated from a seed. The peer Habitat
repo learned this the hard way: without an explicit list its trainer globbed
every scene in the data directory INCLUDING the held-out ones, so a policy
trained on its own eval scenes and nothing in the logs looked wrong.

The defence here is that load_split() has no default. A caller that forgets to
pass a split gets an exception, never "all towns".
"""
import json
import os
from pathlib import Path

DEFAULT_SPLIT = Path(__file__).resolve().parents[2] / "configs" / "town_split.json"


def load_split(path=None):
    """Return (train_towns, heldout_towns) from the frozen JSON."""
    path = Path(path or os.environ.get("TOWN_LIST") or DEFAULT_SPLIT)
    if not path.exists():
        raise SystemExit(
            f"town split not found at {path}. Training must never fall back to "
            "'every town' -- that is how a policy ends up trained on its own "
            "eval maps. Pass TOWN_LIST or restore configs/town_split.json.")
    split = json.loads(path.read_text())
    train, heldout = list(split["train"]), list(split["heldout"])

    overlap = set(train) & set(heldout)
    if overlap:
        raise SystemExit(f"town split is not disjoint: {sorted(overlap)}")
    print(f"[Towns] frozen split from {path}: "
          f"{len(train)} train {train}, {len(heldout)} held out {heldout}")
    return train, heldout


def assert_not_heldout(town, heldout):
    """Guard for the training loop: refuse to train on a held-out town."""
    name = str(town).split("/")[-1].replace(".xodr", "")
    if name in set(heldout):
        raise SystemExit(
            f"refusing to train on held-out town {name}. Held out: {sorted(heldout)}")
    return name
