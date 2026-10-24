import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pytest


@pytest.fixture(autouse=True)
def _tests_never_touch_a_real_history_database(monkeypatch):
    """An AUTOSCALER_DATABASE_URL exported in the shell (the live history database) must not receive rows from the test suite: the tests that need a
    database use TEST_DATABASE_URL, each in a private schema, and set the URL themselves."""
    from src import config
    monkeypatch.setattr(config, "DATABASE_URL", "")
