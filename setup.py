"""Setup shim for legacy pip / build front-ends.

Modern build metadata lives in ``pyproject.toml`` (PEP 621).  This file
exists only for backwards compatibility with tool-chains that still invoke
``setup.py`` directly (e.g. ``pip install .`` on very old pip versions).

It deliberately contains no logic beyond delegating to setuptools.
"""
from setuptools import setup

if __name__ == "__main__":
    setup()
