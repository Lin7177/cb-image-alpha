"""Frozen baseline policy, separate from source locations and experiment artifacts."""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class DataPolicy:
    policy_version: str = "1.1"
    universe: str = "catalog_intersect_historical_identity"
    timestamp_policy: str = "encoded_wall_clock_session"
    timezone: str = "Asia/Shanghai"
    availability_policy: str = "assumed_session_eod"
    decision_clock: str = "23:59:59"
    price_policy: str = "catalog_price_no_coupon"
    volume_policy: str = "catalog_units"
    invalid_bar: str = "fail"
    duplicate_session: str = "fail"


@dataclass(frozen=True)
class SamplePolicy:
    image_days: int = 20
    ma_days: int = 20
    horizon: int = 5
    gap_policy: str = "exclude_no_compression"
    zero_volume_policy: str = "exclude_display_window"
    missing_endpoint_policy: str = "exclude_and_count"
    future_volume_policy: str = "do_not_filter"
    zero_return_label: int = 0


@dataclass(frozen=True)
class ResearchPolicy:
    data: DataPolicy
    samples: SamplePolicy

    @classmethod
    def read(cls, path: Path) -> Self:
        """Reject omissions, unknown fields and changes to the reviewed baseline."""
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        if set(raw) != {"data", "samples"}:
            raise ValueError("Policy must contain exactly data and samples tables")
        for section, expected in (
            ("data", asdict(DataPolicy())), ("samples", asdict(SamplePolicy()))
        ):
            table = raw[section]
            if not isinstance(table, dict) or set(table) != set(expected):
                raise ValueError(f"{section}: policy fields must be complete and exact")
            for name, value in expected.items():
                actual = table[name]
                if type(actual) is not type(value) or actual != value:
                    raise ValueError(f"{section}.{name}: unsupported baseline policy {actual!r}")
        return cls(DataPolicy(**raw["data"]), SamplePolicy(**raw["samples"]))
