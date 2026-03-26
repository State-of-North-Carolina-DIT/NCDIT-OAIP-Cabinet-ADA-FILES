"""API key sanitization for error messages and logs."""
import re

_API_KEY_PATTERNS = [
    re.compile(r'AIzaSy[A-Za-z0-9_-]{33}'),
    re.compile(r'(?i)([\?&]key=)[A-Za-z0-9_-]{20,}'),
    re.compile(r"(?i)(x-goog-api-key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9_-]{20,}"),
    re.compile(r'(?i)(Bearer\s+)[A-Za-z0-9_.-]{20,}'),
]


def sanitize_error(s: str) -> str:
    """Strip API keys from error/exception strings before disk write or logging."""
    for pattern in _API_KEY_PATTERNS:
        s = pattern.sub(lambda m: m.group(0)[:4] + '***REDACTED***', s)
    return s
