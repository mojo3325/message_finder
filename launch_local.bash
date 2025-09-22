#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[message_finder] Checking Docker availability..."
if ! docker info >/dev/null 2>&1; then
    echo "[message_finder] ERROR: Docker daemon is not running!"
    echo "[message_finder] Please start Docker Desktop and try again."
    echo "[message_finder] On macOS, you can start it from Applications or via:"
    echo "[message_finder]   open -a Docker"
    exit 1
fi

echo "[message_finder] Rebuilding image (no cache, pull latest base)..."
docker compose build --no-cache --pull | cat

echo "[message_finder] Starting/Updating containers..."
docker compose up -d | cat

echo "[message_finder] Current image:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedSince}}" | grep '^message_finder\b' || true

echo "[message_finder] Status:"
docker compose ps | cat

echo "[message_finder] Done. Use 'docker compose logs -f' to follow logs."
echo "[message_finder] To start Docker Desktop on macOS:"
echo "[message_finder]   - Open Docker Desktop from Applications"
echo "[message_finder]   - Or run: open -a Docker"

