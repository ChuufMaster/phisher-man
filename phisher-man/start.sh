#!/bin/bash

uv sync

fastapi run 
# /app/.venv/bin/fastapig run app/main.py --port 80 --host 0.0.0.0
/app/.venv/bin/fastapi run app/main.py --port 80 --host 0.0.0.0
