#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

docker compose up -d
if [[ ! -x "$ROOT/backend/.venv/bin/uvicorn" ]]; then
  echo "Backend environment not found. Run ./setup.sh once first." >&2
  exit 1
fi
if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "Frontend dependencies not found. Run ./setup.sh once first." >&2
  exit 1
fi

cd "$ROOT/backend"
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT
cd "$ROOT/frontend"
npm run dev
