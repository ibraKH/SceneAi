"""Local-only request validation shared by the server and lightweight tests."""

from urllib.parse import urlsplit


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_local_origin(origin: str | None) -> bool:
    """Accept normal browser Origins only when they name the local machine."""
    if not origin:
        return False
    try:
        parsed = urlsplit(origin)
        _ = parsed.port  # validates a malformed/out-of-range port
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in LOCAL_HOSTS
        and parsed.username is None
        and parsed.password is None
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
    )
