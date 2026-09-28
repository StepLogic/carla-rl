"""Per-port CARLA quality (CARLA_QUALITY_PORTS), user 2026-09-28."""
import ast
import os

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "train_carla_tqc.py")


def _fn():
    tree = ast.parse(open(SRC).read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "carla_quality_for_port")
    ns = {"os": os}
    exec(compile(ast.Module([node], []), SRC, "exec"), ns)
    return ns["carla_quality_for_port"]


def test_mapping_and_fallbacks():
    f = _fn()
    env = {"CARLA_QUALITY_PORTS": "2000:Low, 2010:Epic", "CARLA_QUALITY": "Low"}
    assert f(2000, env) == "Low" and f(2010, env) == "Epic"
    assert f(2020, env) == "Low"                       # unlisted -> CARLA_QUALITY
    assert f(2010, {}) == "Epic"                       # nothing set -> Epic
    assert f(2000, {"CARLA_QUALITY": "Low"}) == "Low"  # old behaviour
