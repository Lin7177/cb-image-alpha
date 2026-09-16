from pathlib import Path

import pytest

from cb_image_alpha.config import ResearchPolicy

POLICY = Path(__file__).resolve().parents[2] / "configs/data_policy.toml"


def test_reviewed_policy_is_explicit() -> None:
    policy = ResearchPolicy.read(POLICY)
    assert policy.data.availability_policy == "assumed_session_eod"
    assert policy.samples.image_days == 20
    assert policy.samples.horizon == 5
    assert policy.samples.future_volume_policy == "do_not_filter"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('horizon = 5', 'horizon = 20'),
        ('image_days = 20', 'image_days = true'),
        ('volume_policy = "catalog_units"', 'volume_policy = "shares"'),
        ('decision_clock = "23:59:59"', 'decision_clock = "15:00:00"'),
        ('zero_return_label = 0', ''),
        ('[data]', '[data]\nunknown = "x"'),
    ],
)
def test_unreviewed_contract_changes_are_rejected(tmp_path: Path, old: str, new: str) -> None:
    path = tmp_path / "policy.toml"
    path.write_text(POLICY.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        ResearchPolicy.read(path)
