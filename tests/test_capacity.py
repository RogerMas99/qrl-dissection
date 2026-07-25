"""NEW-02: parameter-budget matching."""
import pytest

from conftest import load_by_path

match_hidden_width = load_by_path(
    "_capacity", "src/qrl_dissection/core/capacity.py").match_hidden_width


def test_matched_width_spends_close_to_budget():
    width, spent = match_hidden_width(target_params=120, in_dim=3, out_dim=2)
    assert width >= 1
    assert abs(spent - 120) <= max(3, 0.1 * 120)


def test_width_grows_with_budget():
    small, _ = match_hidden_width(120, in_dim=3, out_dim=2)
    large, _ = match_hidden_width(1200, in_dim=3, out_dim=2)
    assert large > small


def test_rejects_nonsense_budget():
    with pytest.raises(ValueError):
        match_hidden_width(0, in_dim=3, out_dim=2)
