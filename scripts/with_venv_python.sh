#!/usr/bin/env bash
# Thin wrapper: just run the command as-is.
# NumPy must be installed into the Python that PyO3 links against
# (typically the system Python on CI, venv Python locally).
# The Makefile sync target ensures the venv exists for uv-based tasks.
exec "$@"
