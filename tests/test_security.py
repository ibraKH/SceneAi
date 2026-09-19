import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from security import is_local_origin  # noqa: E402


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8000",
        "https://localhost",
        "http://127.0.0.1:8123",
        "http://[::1]:8000",
    ],
)
def test_local_browser_origins_are_allowed(origin):
    assert is_local_origin(origin)


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "null",
        "https://example.com",
        "http://localhost.example.com",
        "http://user@localhost:8000",
        "http://localhost:99999",
        "file://localhost",
    ],
)
def test_missing_malformed_and_remote_origins_are_rejected(origin):
    assert not is_local_origin(origin)
