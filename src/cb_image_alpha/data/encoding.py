"""Exact decoding of the catalog's explicitly signed fixed-point format."""

from decimal import Decimal


def decode_fixed_i128(value: bytes) -> Decimal:
    """Decode signed little-endian i128 with 16 fractional digits, exactly."""
    if not isinstance(value, bytes) or len(value) != 16:
        raise ValueError("expected exactly 16 bytes for raw_fixed_i128_le")
    integer = int.from_bytes(value, byteorder="little", signed=True)
    digits = tuple(int(char) for char in str(abs(integer)))
    return Decimal((int(integer < 0), digits, -16))
