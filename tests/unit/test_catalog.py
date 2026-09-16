from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cb_image_alpha.data.catalog import CatalogError, read_bars


def write_file(tmp_path, *, metadata_change=None, rows=None, nullable=False, width=16):
    # Fixture specifies the source contract independently of production constants.
    metadata = {
        "instrument_id": "110031.SH.VA",
        "bar_type": "110031.SH.VA-1-DAY-LAST-EXTERNAL",
        "ohlcv_binary_width": "16",
        "ohlcv_encoding": "raw_fixed_i128_le",
        "raw_fixed_precision": "16",
        "raw_fixed_width": "16",
        "raw_fixed_encoding": "signed_twos_complement_le",
        "raw_encoding_version": "1",
        "raw_encoding_profile": "nautilus_i128_v1",
        "price_precision": "3",
        "size_precision": "0",
    }
    metadata.update(metadata_change or {})
    schema = pa.schema(
        [
            *[
                pa.field(name, pa.binary(width), nullable=nullable)
                for name in ("open", "high", "low", "close", "volume")
            ],
            pa.field("ts_event", pa.uint64(), nullable=False),
            pa.field("ts_init", pa.uint64(), nullable=False),
        ],
        metadata=metadata,
    )
    if rows is None:
        rows = [(100, 110, 90, 105, 0, 1), (101, 111, 91, 106, 20, 2)]
    data = []
    for o, h, low, c, v, ts in rows:
        raw = {
            name: (
                int(value * 10**16).to_bytes(width, "little", signed=True)
                if value is not None
                else None
            )
            for name, value in zip(
                ("open", "high", "low", "close", "volume"), (o, h, low, c, v), strict=True
            )
        }
        data.append({**raw, "ts_event": ts, "ts_init": ts})
    path = tmp_path / "bars.parquet"
    pq.write_table(pa.Table.from_pylist(data, schema=schema), path)
    return path


def test_values_and_read_has_no_side_effects(tmp_path):
    path = write_file(tmp_path)
    before = (path.read_bytes(), path.stat().st_mtime_ns, sorted(tmp_path.iterdir()))
    bars = read_bars(path)
    assert len(bars) == 2
    assert bars[0].instrument_id == "110031.SH.VA"
    assert (bars[0].open, bars[0].high, bars[0].low, bars[0].close, bars[0].volume) == (
        Decimal(100),
        Decimal(110),
        Decimal(90),
        Decimal(105),
        Decimal(0),
    )
    assert bars[1].ts_event == bars[1].ts_init == 2
    assert before == (path.read_bytes(), path.stat().st_mtime_ns, sorted(tmp_path.iterdir()))


@pytest.mark.parametrize(
    "key,value",
    [
        ("raw_fixed_precision", "9"),
        ("raw_encoding_profile", "other"),
        ("raw_fixed_encoding", "unsigned_le"),
        ("ohlcv_encoding", "float"),
        ("bar_type", "110031.SH.VA-1-MINUTE-LAST-EXTERNAL"),
        ("instrument_id", "wrong"),
        ("instrument_id", ""),
    ],
)
def test_bad_metadata(tmp_path, key, value):
    with pytest.raises(CatalogError, match="metadata|bar_type"):
        read_bars(write_file(tmp_path, metadata_change={key: value}))


@pytest.mark.parametrize(
    "rows,match",
    [
        ([], "empty"),
        ([(0, 110, 90, 105, 1, 1)], "open must be positive"),
        ([(-1, 110, 90, 105, 1, 1)], "positive"),
        ([(100, 110, 90, 105, -1, 1)], "volume"),
        ([(100, 99, 90, 105, 1, 1)], "OHLC"),
        ([(100, 110, 101, 105, 1, 1)], "OHLC"),
        ([(100, 110, 90, 105, 1, 2), (100, 110, 90, 105, 1, 1)], "increasing"),
        ([(100, 110, 90, 105, 1, 1), (100, 110, 90, 105, 1, 1)], "increasing"),
    ],
)
def test_invalid_rows(tmp_path, rows, match):
    with pytest.raises(CatalogError, match=match):
        read_bars(write_file(tmp_path, rows=rows))


def test_nullable_schema_rejected(tmp_path):
    with pytest.raises(CatalogError, match="schema"):
        read_bars(write_file(tmp_path, nullable=True, rows=[(None, 110, 90, 105, 1, 1)]))


def test_wrong_width_rejected(tmp_path):
    with pytest.raises(CatalogError, match="schema"):
        read_bars(write_file(tmp_path, width=17))


def test_missing_source_does_not_create_it(tmp_path):
    path = tmp_path / "missing" / "file.parquet"
    with pytest.raises(CatalogError, match="file.parquet"):
        read_bars(path)
    assert not path.parent.exists()


def instrument_directory(tmp_path, rows=None, name="110031.SH.VA-1-DAY-LAST-EXTERNAL"):
    directory = tmp_path / name
    directory.mkdir()
    if rows is None:
        rows = [(100, 110, 90, 105, 1, 54_000_000_000_000)]
    write_file(directory, rows=rows)
    return directory


def test_instrument_directory_contract(tmp_path):
    from cb_image_alpha.data.catalog import read_instrument

    directory = instrument_directory(tmp_path)
    assert len(read_instrument(directory)) == 1


@pytest.mark.parametrize("extra", [False, True])
def test_instrument_requires_single_file(tmp_path, extra):
    from cb_image_alpha.data.catalog import read_instrument

    directory = instrument_directory(tmp_path)
    if extra:
        (directory / "second.parquet").write_bytes((directory / "bars.parquet").read_bytes())
    else:
        (directory / "bars.parquet").unlink()
    with pytest.raises(CatalogError, match="exactly one"):
        read_instrument(directory)


def test_directory_identity_mismatch(tmp_path):
    from cb_image_alpha.data.catalog import read_instrument

    directory = instrument_directory(tmp_path, name="wrong")
    with pytest.raises(CatalogError, match="directory identity"):
        read_instrument(directory)


@pytest.mark.parametrize("timestamp", [54_000_000_000_001, 53_999_999_999_999, 0])
def test_daily_marker_exact_to_nanosecond(tmp_path, timestamp):
    from cb_image_alpha.data.catalog import read_instrument

    directory = instrument_directory(tmp_path, rows=[(100, 110, 90, 105, 1, timestamp)])
    with pytest.raises(CatalogError, match="exact UTC 15"):
        read_instrument(directory)


def test_duplicate_encoded_date_even_with_distinct_timestamps(tmp_path):
    from cb_image_alpha.data.catalog import read_instrument

    directory = instrument_directory(
        tmp_path,
        rows=[
            (100, 110, 90, 105, 1, 54_000_000_000_000),
            (100, 110, 90, 105, 1, 54_000_000_000_001),
        ],
    )
    with pytest.raises(CatalogError, match="duplicate encoded date"):
        read_instrument(directory)
