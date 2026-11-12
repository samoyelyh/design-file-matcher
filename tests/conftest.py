"""Never let TestClient startup or background workers connect to live data."""
import os
import tempfile
from pathlib import Path

import pytest


_test_database = tempfile.TemporaryDirectory(prefix="design-matcher-tests-")
os.environ["DESIGN_MATCHER_DB_URL"] = f"sqlite:///{(Path(_test_database.name) / 'matcher.db').as_posix()}"


@pytest.fixture(scope="session", autouse=True)
def isolated_default_database():
    yield
    from app.db import engine

    engine.dispose()
    _test_database.cleanup()
