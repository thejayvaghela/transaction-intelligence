"""Smoke test: the package imports and exposes a version. Proves the env + src-layout work."""

import transaction_intelligence as ti


def test_package_imports():
    assert ti.__version__
