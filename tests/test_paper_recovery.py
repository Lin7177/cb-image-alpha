"""Synthetic fail-closed checks for the authorized range recovery."""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('recover_paper_data', SCRIPTS / 'recover_paper_data.py')
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def headers():
    return {'etag': MODULE.ETAG,
            'content-range': 'bytes 550598230-567446779/567446780',
            'content-length': '16848550'}


def test_exact_response():
    MODULE.validate_response('HTTP/1.1 206 Partial Content', headers())


@pytest.mark.parametrize('status', ['HTTP/1.1 200 OK', 'HTTP/1.1 302 Found', ''])
def test_not_partial(status):
    with pytest.raises(ValueError):
        MODULE.validate_response(status, headers())


@pytest.mark.parametrize(('key', 'value'), [
    ('etag', 'changed'), ('content-range', 'bytes 0-567446779/567446780'),
    ('content-range', 'bytes 550598230-567446778/567446779'),
    ('content-length', '567446780'), ('content-encoding', 'gzip'),
])
def test_changed_object_or_range(key, value):
    response = headers()
    response[key] = value
    with pytest.raises(ValueError):
        MODULE.validate_response('HTTP/1.1 206 Partial Content', response)