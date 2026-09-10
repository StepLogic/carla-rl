"""The placement table. Same reasoning as test_four_way_summary.py: an n/a
printed as 0.000 reads as "this arm is worse" rather than "this number does not
apply", and a table whose arms ran at different budgets is not a comparison at
all while still looking like one.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summarize_placement import format_row, render  # noqa: E402


def _row(method, band="5-20m", budget=40, **kw):
    r = {"method": method, "band": band, "budget": budget, "n_nodes": budget,
         "n_edges": 100, "success": 0.8, "spl": 0.6, "leg_timeouts": 2,
         "end_reasons": {"arrived": 48, "no_path": 3}}
    r.update(kw)
    return r


ALL_FOUR = [_row(m) for m in ("uniform", "random", "policy", "sgm")]


def _methods(out):
    names = set(("uniform", "random", "policy", "sgm"))
    return [ln.split()[0] for ln in out.splitlines()
            if ln and ln.split()[0] in names]


def test_arms_are_ordered_baselines_first():
    # Reading down the column is then "what does consulting the policy buy".
    assert _methods(render(ALL_FOUR)) == ["uniform", "random", "policy", "sgm"]


def test_ordering_is_imposed_not_inherited():
    assert _methods(render(list(reversed(ALL_FOUR)))) == [
        "uniform", "random", "policy", "sgm"]


def test_a_missing_spl_prints_na_not_zero():
    # A zero would read as "took an infinite detour" rather than "was never
    # measured", which is the difference between a bad result and no result.
    assert "n/a" in format_row(_row("sgm", spl=None))


def test_unroutable_episodes_are_counted_from_the_reasons():
    row = format_row(_row("uniform", end_reasons={"arrived": 10, "no_path": 4,
                                                  "start_unlocalised": 2}))
    assert " 6 " in row or row.rstrip().endswith("6       2")


def test_different_budgets_are_reported_loudly():
    # It silently invalidates the comparison: a difference between arms may be
    # sparsity rather than placement.
    out = render([_row("uniform", budget=40), _row("sgm", budget=90)])
    assert "DIFFERENT budgets" in out


def test_one_budget_does_not_warn():
    assert "DIFFERENT budgets" not in render(ALL_FOUR)


def test_a_band_beyond_the_training_range_is_labelled_extrapolation():
    # It is the TOP of the band that decides this, not the bottom.
    assert "EXTRAPOLATION" in render([_row("sgm", band="20-40m")])
    assert "in-distribution" in render([_row("sgm", band="5-20m")])


def test_a_partial_table_says_which_arms_are_missing():
    out = render([_row("uniform"), _row("sgm")])
    assert "MISSING" in out and "policy" in out.split("MISSING")[1]


def test_an_empty_directory_says_no_results():
    assert render([]) == "no results found"
