"""BC_SHARD_LIST (plan 2026-11-09, B3): the clone c1 trains on M1 (il_data, seeds
<= 199999, as c0) plus the DAgger-in-chains shards (il_data_dagger_chain1, ids
seed*100+segment) -- a per-root rule BC_MAX_SEED cannot express. An explicit list
file (one shard path per line) selects the data; it is recorded in bc_split.json.
Held-out towns are refused, as for a root. Unset: the root + BC_MAX_SEED path as
before.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.bc_trainer import bc_episode_paths  # noqa: E402


def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()
    return str(p)


def test_without_a_list_the_root_and_max_seed_rule_is_unchanged(tmp_path):
    a = _touch(tmp_path / "il" / "Town05" / "ep_100.npz")
    _touch(tmp_path / "il" / "Town05" / "ep_300000.npz")
    assert bc_episode_paths(str(tmp_path / "il"), ["Town05"], max_seed="199999") == [a]


def test_a_list_selects_exactly_its_shards(tmp_path):
    a = _touch(tmp_path / "il" / "Town05" / "ep_100.npz")
    b = _touch(tmp_path / "chain" / "Town01" / "ep_90000000.npz")
    lst = tmp_path / "list.txt"
    lst.write_text(f"{a}\n\n{b}\n")
    got = bc_episode_paths("", ["Town05", "Town01"], max_seed="199999", shard_list=str(lst))
    assert got == sorted([a, b])                     # the seed cap does not apply to a list


def test_a_list_refuses_missing_files_and_held_out_or_unlisted_towns(tmp_path):
    lst = tmp_path / "list.txt"
    lst.write_text(str(tmp_path / "il" / "Town05" / "ep_1.npz") + "\n")
    with pytest.raises(SystemExit):
        bc_episode_paths("", ["Town05"], shard_list=str(lst))          # missing file
    held = _touch(tmp_path / "il" / "Town02" / "ep_1.npz")
    lst.write_text(held + "\n")
    with pytest.raises((SystemExit, ValueError)):
        bc_episode_paths("", ["Town02"], shard_list=str(lst))          # held-out town
    other = _touch(tmp_path / "il" / "Town04" / "ep_1.npz")
    lst.write_text(other + "\n")
    with pytest.raises(SystemExit):
        bc_episode_paths("", ["Town05"], shard_list=str(lst))          # not a BC_TOWNS town
