"""Every name main() reads as a global must exist (2026-11-05).

build_agent() took over building the learner's config (imitation-first plan,
Task 5), but main() kept printing cfg.lr, cfg.mode_index and the rest. Cloning,
eval and recording all return before those lines, so the first training run to
reach them -- the fine-tune, job 463421 -- died with "NameError: name 'cfg' is
not defined" after seven minutes of start-up (four CARLA servers, the warm
start). main() cannot run under test, but the names it reads can be checked.
"""
import builtins
import dis
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def _global_loads(code):
    for ins in dis.get_instructions(code):
        if ins.opname == "LOAD_GLOBAL":
            yield ins.argval
    for const in code.co_consts:
        if hasattr(const, "co_code"):          # nested functions and lambdas
            yield from _global_loads(const)


def test_main_reads_no_undefined_global():
    import train_carla_sac as m
    missing = sorted({name for name in _global_loads(m.main.__code__)
                      if not hasattr(m, name) and not hasattr(builtins, name)})
    assert missing == [], f"main() reads undefined names: {missing}"
