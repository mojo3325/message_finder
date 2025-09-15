#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[message_finder] Rebuilding image (no cache, pull latest base)..."
docker compose build --no-cache --pull | cat

echo "[message_finder] Starting/Updating containers..."
docker compose up -d | cat

echo "[message_finder] Current image:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedSince}}" | grep '^message_finder\b' || true

echo "[message_finder] Status:"
docker compose ps | cat

echo "[message_finder] Done. Use 'docker compose logs -f' to follow logs."

