from fastapi.testclient import TestClient

import kite_client
import main


def test_login_url_uses_kiteconnect_supported_signature(monkeypatch):
    class FakeKiteConnect:
        def __init__(self, api_key):
            self.api_key = api_key

        def login_url(self):
            return f"https://kite.test/login?api_key={self.api_key}"

    monkeypatch.setattr(kite_client, "KiteConnect", FakeKiteConnect)
    monkeypatch.setenv("KITE_API_KEY", "test-key")

    assert kite_client.get_login_url() == "https://kite.test/login?api_key=test-key"


def test_kite_callback_saves_session_without_returning_token(monkeypatch):
    saved_tokens = []
    monkeypatch.setattr(main, "generate_access_token", lambda request_token: "daily-token")
    monkeypatch.setattr(main, "save_kite_access_token", saved_tokens.append)
    monkeypatch.setenv("FRONTEND_URL", "http://localhost:5173")

    response = TestClient(main.app).get("/callback?request_token=request-token")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "location" not in response.headers
    assert "daily-token" not in response.text
    assert saved_tokens == ["daily-token"]


def test_kite_callback_returns_to_dashboard_only_with_login_cookie(monkeypatch):
    monkeypatch.setattr(main, "generate_access_token", lambda request_token: "daily-token")
    monkeypatch.setattr(main, "save_kite_access_token", lambda token: None)

    client = TestClient(
        main.app,
        follow_redirects=False,
        cookies={main.RETURN_COOKIE: "http://localhost:5173/"},
    )
    response = client.get("/auth/kite/callback?request_token=request-token")

    assert response.status_code == 303
    assert response.headers["location"] == "http://localhost:5173/?kite=success"


def test_holdings_sync_requires_kite_login(monkeypatch):
    monkeypatch.setattr(main, "get_authenticated_kite", lambda: None)

    response = TestClient(main.app).post("/sync/holdings")

    assert response.status_code == 401
    assert response.json()["error"] == "Not logged in. Visit /login first."