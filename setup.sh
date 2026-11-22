#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
docker compose up -d
cd "$ROOT/backend"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
python scripts/prepare_and_index.py
python scripts/verify_index.py
cd "$ROOT/frontend"
[ -f .env.local ] || cp .env.example .env.local
npm install
printf '\nStart the API in another terminal:\n  cd backend && source .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000\n\nThen run the frontend:\n  cd frontend && npm run dev\n'
