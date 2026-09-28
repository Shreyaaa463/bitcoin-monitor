import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.helpers import generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory) -> Path:
    return generate_dataset(tmp_path_factory.mktemp("data") / "small.json", n=600, seed=7)
