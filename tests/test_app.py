"""Smoke test: the Streamlit app renders every tab without errors."""
from streamlit.testing.v1 import AppTest


def test_app_runs():
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 4
