import pytest

from frontend_redirect import build_redirect_url, safe_return_url


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("http://localhost:5173/", "http://localhost:5173/"),
        ("http://localhost:3000/dashboard?x=1#frag", "http://localhost:3000/dashboard"),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080/"),
        ("http://[::1]:5173/", "http://[::1]:5173/"),
        ("https://localhost/", "https://localhost/"),
    ],
)
def test_local_return_urls_are_accepted_and_normalised(candidate, expected):
    assert safe_return_url(candidate) == expected


@pytest.mark.parametrize(
    "candidate",
    [
        None,
        "",
        "https://evil.com/",
        "http://localhost.evil.com/",
        "http://evil.com/localhost",
        "http://localhost@evil.com/",
        "http://evil.com\\@localhost/",
        "//evil.com/",
        "/relative/path",
        "javascript:alert(1)",
        "ftp://localhost/",
        "http://localhost:notaport/",
        "http://localhost/\r\nSet-Cookie: x=y",
    ],
)
def test_foreign_or_malformed_return_urls_are_rejected(candidate):
    assert safe_return_url(candidate) is None


def test_configured_frontend_origin_is_allowed(monkeypatch):
    monkeypatch.setenv("FRONTEND_URL", "https://meridian.example.com")
    assert safe_return_url("https://meridian.example.com/app") == "https://meridian.example.com/app"
    assert safe_return_url("https://other.example.com/app") is None


def test_build_redirect_url_adds_status_without_secrets():
    url = build_redirect_url("http://localhost:5173/", "success")
    assert url == "http://localhost:5173/?kite=success"
    err = build_redirect_url("http://localhost:5173/", "error", "Token expired & invalid")
    assert err.startswith("http://localhost:5173/?kite=error&message=Token+expired")
    long = build_redirect_url("http://localhost:5173/", "error", "x" * 500)
    assert len(long) < 300
