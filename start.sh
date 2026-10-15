#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] && [ ! -f .venv/Scripts/python.exe ]; then
    if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
        python3 -m venv .venv
    elif command -v python >/dev/null 2>&1 && python -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
        python -m venv .venv
    else
        echo 'Python 3.11+ is required. Install it and add it to PATH.' >&2
        exit 1
    fi
fi
if [ -f .venv/Scripts/python.exe ]; then
    venv_python=.venv/Scripts/python.exe
else
    venv_python=.venv/bin/python
fi
# pip skips already satisfied dependencies on subsequent starts.
"$venv_python" -m pip install -r requirements.txt
exec "$venv_python" -m uvicorn main:app --reload --port 8000
