"""Smoke test for the Streamlit UI: it must render, and fail gracefully when the API is down."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

UI = str(Path(__file__).parents[1] / "ui.py")


def test_ui_renders_and_reports_unreachable_api(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens on port 9
    app = AppTest.from_file(UI, default_timeout=30).run()
    assert not app.exception
    assert any("API not reachable" in e.value for e in app.error)
    assert any("Upload a workbook" in i.value for i in app.info)
