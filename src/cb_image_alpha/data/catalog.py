"""Read-only catalog decoding; no calendar or economic-unit assumptions."""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cb_image_alpha.data.encoding import decode_fixed_i128

PRICE_FIELDS = ("open", "high", "low", "close")
VALUE_FIELDS = (*PRICE_FIELDS, "volume")
ENCODING_METADATA = {
    b"ohlcv_binary_width": b"16",
    b"ohlcv_encoding": b"raw_fixed_i128_le",
    b"raw_fixed_precision": b"16",
    b"raw_fixed_width": b"16",
    b"raw_fixed_encoding": b"signed_twos_complement_le",
    b"raw_encoding_version": b"1",
    b"raw_encoding_profile": b"nautilus_i128_v1",
    b"price_precision": b"3",
    b"size_precision": b"0",
}


class CatalogError(ValueError):
    """A catalog file does not satisfy the verified raw-format contract."""


@dataclass(frozen=True)
class RawBar:
    instrument_id: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    ts_event: int
    ts_init: int


def _validate_schema(schema: pa.Schema) -> str:
    expected = [pa.field(name, pa.binary(16), nullable=False) for name in VALUE_FIELDS]
    expected += [pa.field(name, pa.uint64(), nullable=False) for name in ("ts_event", "ts_init")]
    if not schema.remove_metadata().equals(pa.schema(expected)):
        raise CatalogError("schema must contain non-null OHLCV binary[16] and uint64 timestamps")
    metadata = schema.metadata or {}
    for key, value in ENCODING_METADATA.items():
        if metadata.get(key) != value:
            raise CatalogError(f"metadata {key.decode()} must equal {value.decode()}")
    try:
        instrument = metadata[b"instrument_id"].decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise CatalogError("missing or invalid instrument_id metadata") from exc
    if not instrument or instrument.strip() != instrument:
        raise CatalogError("invalid instrument_id metadata")
    if metadata.get(b"bar_type") != f"{instrument}-1-DAY-LAST-EXTERNAL".encode():
        raise CatalogError("bar_type must match instrument_id and 1-DAY-LAST-EXTERNAL")
    return instrument


def read_bars(path: Path) -> list[RawBar]:
    """Read one file, rejecting invalid rows; never creates directories or outputs."""
    try:
        with pq.ParquetFile(path) as parquet:
            instrument = _validate_schema(parquet.schema_arrow)
            table = parquet.read()
        if table.num_rows == 0:
            raise CatalogError("empty bar file")
        if any(column.null_count for column in table.columns):
            raise CatalogError(f"{instrument}: null bar values")
        bars = []
        previous = -1
        for index, row in enumerate(table.to_pylist()):
            values = {field: decode_fixed_i128(row[field]) for field in VALUE_FIELDS}
            context = f"{instrument} row {index}"
            for field in PRICE_FIELDS:
                if values[field] <= 0:
                    raise CatalogError(f"{context}: {field} must be positive")
            if values["volume"] < 0:
                raise CatalogError(f"{context}: volume must be nonnegative")
            if not (
                values["low"]
                <= min(values["open"], values["close"])
                <= max(values["open"], values["close"])
                <= values["high"]
            ):
                raise CatalogError(f"{context}: invalid OHLC relation")
            if row["ts_event"] <= previous:
                raise CatalogError(f"{context}: ts_event must be strictly increasing")
            previous = row["ts_event"]
            bars.append(
                RawBar(
                    instrument_id=instrument,
                    **values,
                    ts_event=row["ts_event"],
                    ts_init=row["ts_init"],
                )
            )
        return bars
    except (CatalogError, ValueError, OSError, pa.ArrowException) as exc:
        raise CatalogError(f"{path}: {exc}") from exc


def read_instrument(directory: Path) -> list[RawBar]:
    """Read the single-file daily directory contract, without interpreting availability.

    The encoded UTC date and exact 15:00 marker are source-format assertions;
    neither the marker nor ts_init is asserted to be an economic availability time.
    """
    if not directory.is_dir():
        raise CatalogError(f"{directory}: expected instrument directory")
    files = sorted(directory.glob("*.parquet"))
    if len(files) != 1:
        raise CatalogError(f"{directory}: expected exactly one parquet file, found {len(files)}")
    bars = read_bars(files[0])
    expected_name = f"{bars[0].instrument_id}-1-DAY-LAST-EXTERNAL"
    if directory.name != expected_name:
        raise CatalogError(f"{directory}: directory identity must equal {expected_name}")
    day_ns = 86_400 * 1_000_000_000
    marker_ns = 15 * 3_600 * 1_000_000_000
    seen_dates: set[int] = set()
    for index, bar in enumerate(bars):
        encoded_day, time_ns = divmod(bar.ts_event, day_ns)
        if encoded_day in seen_dates:
            raise CatalogError(f"{directory}: row {index}: duplicate encoded date")
        seen_dates.add(encoded_day)
        if time_ns != marker_ns:
            raise CatalogError(f"{directory}: row {index}: ts_event must mark exact UTC 15:00:00")
    return bars
