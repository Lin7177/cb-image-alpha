from decimal import Decimal, localcontext

import pytest

from cb_image_alpha.data.encoding import decode_fixed_i128


@pytest.mark.parametrize(
    "integer,expected",
    [
        (0, "0"),
        (1235000000000000000, "123.5"),
        (-1, "-0.0000000000000001"),
        (2**127 - 1, "17014118346046923173168.7303715884105727"),
        (-(2**127), "-17014118346046923173168.7303715884105728"),
    ],
)
def test_exact_decode_independent_of_decimal_context(integer, expected):
    with localcontext() as context:
        context.prec = 2
        assert decode_fixed_i128(integer.to_bytes(16, "little", signed=True)) == Decimal(expected)


@pytest.mark.parametrize("value", [None, 1, bytearray(16), b"", bytes(15), bytes(17)])
def test_bad_bytes_rejected(value):
    with pytest.raises(ValueError, match="16 bytes"):
        decode_fixed_i128(value)
