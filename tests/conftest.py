"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from scripts.core.data import prepare_data


@pytest.fixture(scope="session")
def bundle():
    """One prepared dataset bundle, reused by every test that needs real data."""
    return prepare_data()
