"""Keep os.environ changes from leaking between tests.

Several tests drive a script's `main()`, and those `main()`s call
`load_dotenv(backfill_data/.env)` — which is correct for the script and toxic
for the suite: it puts real provider keys into `os.environ` for every test
that runs afterwards. `test_provider_requires_api_key` then finds a key where
it asserted there was none, so it passes alone and fails in a full run purely
on collection order.

Snapshot-and-restore around every test, rather than deleting specific keys, so
this keeps holding when the next script learns to read a new variable.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_env():
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)
