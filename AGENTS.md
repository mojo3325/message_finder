# Repository Guidelines

## Project Structure & Module Organization
- Entry points: `message_finder.py` (listener) and `warp_chat.py` (bot utilities).
- Core libs: `core/` (dedup, rate limiting, types), `services/` (LLM clients, classification, replies), `tg/` (Telegram API, UI, poller), `utilities/` (storage helpers), `utils/` (misc helpers).
- Config & logging: `config.py` (env-driven settings), `logging_config.py` (JSON logs).
- Data & outputs: `data/` (runtime state, e.g., accounts), `results/` (CSV/exports).

## Build, Test, and Development Commands
- Setup (local):
  - `python -m venv venv && source venv/bin/activate`
  - `pip install -r requirements.txt`
- Run (local):
  - Listener: `python message_finder.py`
  - Warp Chat: `python warp_chat.py`
  - Limit scope for testing: `TEST_ONLY_CHAT_ID=<chat_id> python message_finder.py`
- Docker:
  - `docker compose up -d` (or `./launch_local.bash` to rebuild + start)
  - Logs: `docker compose logs -f message_finder`

## Coding Style & Naming Conventions
- Python 3.11, 4-space indentation, type hints required for new code.
- Use snake_case for files, functions, and variables; PascalCase for classes.
- Prefer dependency-free helpers in `core/`; external-service logic in `services/`; Telegram-related code in `tg/`.
- Use `logging_config.logger` (structured JSON). Avoid `print`.
- Keep modules small and cohesive; extract pure logic for easy testing.

## Testing Guidelines
- Framework: pytest (add to your venv: `pip install pytest`).
- Location: create `tests/` with files named `test_<module>.py`.
- Focus: unit-test pure logic in `core/` and service boundaries with fakes.
- Run: `pytest -q`. Optional coverage: `pytest -q --maxfail=1 --disable-warnings`.

## Commit & Pull Request Guidelines
- Prefer Conventional Commits where practical: `feat:`, `fix:`, `chore:`, `refactor:`, `docs:`.
- Commits should be small, with clear intent and scoped diffs.
- PRs must include: purpose, summary of changes, runtime/test notes, and any config/env impacts. Link issues when relevant.

## Security & Configuration Tips
- Do not commit secrets. Override via env vars (see `config.py`), e.g. `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `LMSTUDIO_BASE_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `TELEGRAM_BOT_TOKEN`.
- For local-only experiments, use a `.env` file or shell exports; rotate any leaked credentials.
- Rate limits are configurable via env (e.g., `RATE_LIMIT_RPM`, `GEMINI_RATE_*`).

## Agent-Specific Instructions
- Follow this guide and keep changes minimal and surgical.
- Respect existing module boundaries and logging; avoid unrelated refactors.
- Add notes in PRs about any behavior or config changes.
