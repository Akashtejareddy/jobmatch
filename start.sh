#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
# pip skips already satisfied dependencies on subsequent starts.
.venv/bin/python -m pip install -r requirements.txt
exec .venv/bin/python -m uvicorn main:app --reload --port 8000
