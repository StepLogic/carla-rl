"""The four-way comparison table.

WHY A TABLE FORMATTER GETS TESTS. It is the deliverable of the whole exercise
and the only artefact anyone will read. The failure modes are not crashes: an
n/a printed as 0.000 reads as "this arm is worse" rather than "this number does
not apply to this arm", and a partial table read as a complete one supports a
four-way claim on three arms. Both are silent, and both survive any amount of
staring at the code that produced them.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summarize_four_way import format_row, load, render  # noqa: E402


def _arm(arm, goal_conditioned=True, success=0.9, steps_median=90.0,
         n_arrived=54, episode_len_mean=120.0, timeout_frac=0.05,
         q_bias_steps=-12.0, q_corr=0.81, q_n=54):
    return {"arm": arm, "goal_conditioned": goal_conditioned,
            "success": success if goal_conditioned else None,
            "steps_median": steps_median, "n_arrived": n_arrived,
            "episode_len_mean": episode_len_mean, "timeout_frac": timeout_frac,
            "q_bias_steps": q_bias_steps, "q_corr": q_corr, "q_n": q_n}


ALL_FOUR = [_arm("goal_only"), _arm("goal_free", goal_conditioned=False,
                                    steps_median=None, n_arrived=0),
            _arm("gated"), _arm("gated_her")]


def test_the_goal_free_arm_shows_n_a_not_a_zero():
    # THE LOAD-BEARING ONE. 0.000 in this column reads as "worse than the
    # others" when the truth is "this metric does not apply".
    row = format_row(_arm("goal_free", goal_conditioned=False,
                          steps_median=None, n_arrived=0))
    assert "n/a" in row
    assert "0.000" not in row


def test_a_goal_conditioned_arm_shows_its_success():
    assert "0.900" in format_row(_arm("gated", success=0.9))


def test_steps_show_n_a_when_nothing_arrived():
    # Asserts the STEPS column specifically. Counting "n/a" occurrences was
    # brittle: it broke the moment a column was added, for no real reason.
    row = format_row(_arm("gated", steps_median=None, n_arrived=0))
    assert row.split()[2] == "n/a"


def _arm_column(out):
    """The arm names in table order. Parsed from the first column so that
    'gated' matching inside 'gated_her' cannot make a wrong order look right."""
    names = {"goal_only", "goal_free", "gated", "gated_her"}
    return [line.split()[0] for line in out.splitlines()
            if line and line.split()[0] in names]


def test_arms_are_ordered_specialists_first():
    assert _arm_column(render(ALL_FOUR)) == [
        "goal_only", "goal_free", "gated", "gated_her"]


def test_ordering_is_imposed_not_inherited_from_input():
    # render() must sort, not trust its caller: rows arrive from tests and
    # notebooks as well as from load().
    assert _arm_column(render(list(reversed(ALL_FOUR)))) == [
        "goal_only", "goal_free", "gated", "gated_her"]


def test_a_complete_table_does_not_warn():
    assert "MISSING ARMS" not in render(ALL_FOUR)


def test_a_partial_table_says_so():
    # A four-way claim cannot be made from three arms, and the absence is easy
    # to miss when the table still looks tidy.
    out = render([_arm("goal_only"), _arm("gated")])
    assert "MISSING ARMS" in out
    assert "goal_free" in out.split("MISSING ARMS")[1]
    assert "gated_her" in out.split("MISSING ARMS")[1]


def test_an_empty_directory_says_no_results():
    assert render([]) == "no results found"


def test_the_footnotes_explain_the_n_a_and_the_steps_column():
    out = render(ALL_FOUR)
    assert "BY CONSTRUCTION" in out
    assert "negated return" in out
    assert "SAME episode set" in out


def test_load_reads_and_orders_files(tmp_path):
    for row in reversed(ALL_FOUR):
        (tmp_path / f"{row['arm']}.json").write_text(json.dumps(row))
    rows = load(str(tmp_path))
    assert [r["arm"] for r in rows] == ["goal_only", "goal_free", "gated",
                                        "gated_her"]


def test_load_on_an_empty_directory_returns_nothing(tmp_path):
    assert load(str(tmp_path)) == []


# ------------------------------------------------- critic as a distance ----

def test_the_calibration_columns_are_shown():
    # These test the claim the whole reward choice rests on: under minimum time
    # the value IS the negated time to goal.
    row = format_row(_arm("gated", q_bias_steps=-570.0, q_corr=0.42))
    assert "-570" in row
    assert "0.42" in row


def test_a_positive_bias_keeps_its_sign():
    # Sign carries the meaning: negative means the critic thinks the goal is
    # FURTHER than it is. An unsigned number would lose that.
    assert "+18" in format_row(_arm("gated", q_bias_steps=18.0))


def test_calibration_shows_n_a_when_it_could_not_be_measured():
    row = format_row(_arm("goal_free", goal_conditioned=False,
                          steps_median=None, n_arrived=0,
                          q_bias_steps=None, q_corr=None, q_n=0))
    assert row.split()[-1] == "n/a"
    assert row.split()[-2] == "n/a"


def test_the_footnotes_explain_bias_and_correlation_separately():
    # A badly biased critic can still ORDER states correctly, which is all a
    # map builder needs if it compares distances rather than reading them.
    out = render(ALL_FOUR)
    assert "q_bias is the mean signed error IN STEPS" in out
    assert "ORDERS states correctly" in out
    assert "ARRIVING episodes only" in out
