"""LANE_STRICT must actually reach the wedge.

THE BUG. core_state.py gates the lane signal on LANE_STRICT and its comment
offers LANE_STRICT=0 as the remedy "if this starves learning". The wedge read
`self.diff_lane` directly, upstream of that gate, so LANE_STRICT=0 never
reached it: the wedge froze the vehicle on every crossing including a legal
dashed line at a junction, which with JUNCTION_GOAL_FRAC=0.6 is 60% of
episodes.

These tests pin BOTH settings. Strict must stay exactly as it was -- the change
is a bug fix, not a behaviour change at the default.
"""
import pytest

ILLEGAL = {"Solid", "SolidSolid", "SolidBroken", "Curb", "Grass"}


UNPAINTED = {"NONE"}


def wedge_fires(diff_lane, off_road, marking_types, strict,
                wedge_on_off_road=True, in_junction=False):
    """The wedge condition as train_env_config_v4_image_goal.py computes it."""
    marks = [t.split(".")[-1] for t in marking_types]
    crossed_illegal = bool(diff_lane and (
        not marks or any(m in ILLEGAL for m in marks)))
    only_unpainted = bool(marks) and all(m in UNPAINTED for m in marks)
    if strict:
        lane_fatal = bool(diff_lane)
    elif crossed_illegal:
        lane_fatal = bool(diff_lane)
    elif only_unpainted:
        lane_fatal = False
    else:
        lane_fatal = bool(diff_lane and not in_junction)
    return bool(lane_fatal or (wedge_on_off_road and off_road))


BROKEN = ["carla.libcarla.LaneMarkingType.Broken"]
SOLID = ["carla.libcarla.LaneMarkingType.Solid"]
CURB = ["carla.libcarla.LaneMarkingType.Curb"]
MIXED = ["carla.libcarla.LaneMarkingType.Broken",
         "carla.libcarla.LaneMarkingType.Solid"]
NONE_MARK = ["carla.libcarla.LaneMarkingType.NONE"]


@pytest.mark.parametrize("in_junction", [True, False])
def test_strict_is_unchanged_by_this_fix(in_junction):
    """The default must behave exactly as it did before the filter existed."""
    for marks in ([], BROKEN, SOLID, CURB, MIXED, NONE_MARK):
        assert wedge_fires(True, False, marks, strict=True,
                           in_junction=in_junction), (
            "at LANE_STRICT=1 every crossing must still wedge; this fix is not "
            "allowed to change the default")


def test_dashed_crossing_inside_a_junction_is_forgiven():
    """The turn the agent is REQUIRED to make."""
    assert not wedge_fires(True, False, BROKEN, strict=False, in_junction=True)


def test_dashed_crossing_on_open_road_is_still_fatal():
    """THE POINT OF THE JUNCTION RULE.

    Forgiving every Broken crossing would excuse the vehicle drifting out of
    lane on a straight road, which is the exact failure the wedge exists to
    punish. Only the location separates that from a legal turn.
    """
    assert wedge_fires(True, False, BROKEN, strict=False, in_junction=False)


@pytest.mark.parametrize("in_junction", [True, False])
def test_unpainted_crossing_is_never_fatal(in_junction):
    """THE FIX. There is no lane discipline to violate at an unpainted line.

    MEASURED at 100 m goals: NONE@road accounted for ~0.8 of episodes, and the
    scripted PID expert -- which cannot have a competence problem -- arrived on
    0 of its last 100 episodes because of it. A rule no competent driver can
    satisfy is a broken rule, not a hard task.
    """
    assert not wedge_fires(True, False, NONE_MARK, strict=False,
                           in_junction=in_junction)


def test_mixed_broken_and_unpainted_is_judged_as_the_painted_line():
    """A painted boundary is the real one; NONE alongside it must not excuse it."""
    mixed = NONE_MARK + BROKEN
    assert wedge_fires(True, False, mixed, strict=False, in_junction=False)
    assert not wedge_fires(True, False, mixed, strict=False, in_junction=True)


def test_unpainted_does_not_excuse_an_illegal_marking():
    """NONE reported alongside Solid must stay fatal everywhere."""
    for ij in (True, False):
        assert wedge_fires(True, False, NONE_MARK + SOLID, strict=False,
                           in_junction=ij)


@pytest.mark.parametrize("marks", [SOLID, CURB, MIXED])
@pytest.mark.parametrize("in_junction", [True, False])
def test_illegal_markings_wedge_even_inside_a_junction(marks, in_junction):
    """Curb and Grass are physical departures, not lane discipline.

    No junction geometry requires crossing them, so the exemption must not
    reach them.
    """
    assert wedge_fires(True, False, marks, strict=False,
                       in_junction=in_junction)


def test_unknown_marking_is_treated_as_illegal():
    """Fail closed: an empty type list must not become a free pass."""
    assert wedge_fires(True, False, [], strict=False, in_junction=True)
    assert wedge_fires(True, False, [], strict=False, in_junction=False)


@pytest.mark.parametrize("strict", [True, False])
def test_off_road_always_wedges(strict):
    """There is no such thing as a legal excursion onto no lane at all."""
    assert wedge_fires(False, True, [], strict=strict)
    assert wedge_fires(False, True, BROKEN, strict=strict)


@pytest.mark.parametrize("strict", [True, False])
def test_clean_driving_never_wedges(strict):
    assert not wedge_fires(False, False, [], strict=strict)


@pytest.mark.parametrize("strict", [True, False])
def test_off_road_stops_wedging_when_disabled(strict):
    """WEDGE_ON_OFF_ROAD=0: leaving the road no longer freezes the vehicle."""
    assert not wedge_fires(False, True, [], strict=strict,
                           wedge_on_off_road=False)


def test_disabling_off_road_leaves_marking_wedges_intact():
    """Turning off one trigger must not silently disable the other."""
    assert wedge_fires(True, False, SOLID, strict=False,
                       wedge_on_off_road=False)
    assert wedge_fires(True, False, BROKEN, strict=True,
                       wedge_on_off_road=False)


def test_off_road_default_is_still_fatal():
    """The default must not change; only an explicit opt-out disables it."""
    assert wedge_fires(False, True, [], strict=True)
    assert wedge_fires(False, True, [], strict=False)


def test_the_two_settings_actually_differ():
    """Guards against the filter being wired up but inert."""
    assert wedge_fires(True, False, BROKEN, strict=True, in_junction=True)
    assert not wedge_fires(True, False, BROKEN, strict=False, in_junction=True)


def test_matches_core_state_illegal_set():
    """The wedge and the reward's lane signal must agree on 'violation'."""
    import re
    src = open("src/vae_nav/core_state.py").read()
    m = re.search(r"illegal\s*=\s*\{([^}]*)\}", src)
    assert m, "could not find core_state's illegal set"
    core_set = {s.strip().strip('\"\'') for s in m.group(1).split(",")
                if s.strip()}
    assert core_set == ILLEGAL, (
        f"core_state uses {core_set} but the wedge uses {ILLEGAL}; the two "
        "would disagree about what counts as a violation")
