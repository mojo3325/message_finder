### Цель
- Добавить в бота команду `/account` для онбординга пользовательской MTProto-сессии (Telethon) и дальнейшей работы от лица пользователя (читать/отвечать на его сообщения).
-Настроить оркестрацию сразу двумя ботами одновременно. Поменять название message_finder на bot_orchestrator.

### Ключевые принципы
- **Один `API_ID/API_HASH` вашего приложения** для всех пользователей. Не создаём приложение на каждого пользователя.
- **Сессия пользователя изолирована** через его `StringSession`. Ваша текущая “основная” сессия продолжает в реальном времени читать группы — не конфликтует.
- **Безопасное хранение** сессий: минимизация PII, `/unlink` для отзыва.

### Функциональные требования
- **Команды**
  - `/account` — старт онбординга: выбор способа логина (номер+код(+2FA)) или QR.(для обоих ботов)
  - `/unlink` — удалить сессию и разлогинить.(для обоих ботов)
- **Логин: номер+код(+2FA)**
  - Запросить номер (E.164), отправить код, принять код, при необходимости — пароль 2FA.
  - Обработать ошибки: неверный код, срок истёк, FloodWait, 2FA отсутствует/нужен.
- **Логин: QR**
  - Сгенерировать QR через Telethon, выслать пользователю (изображение/ASCII), ждать подтверждения, сохранить сессию.
- **Сохранение**
  - По завершении логина — получить `StringSession`, сохранить в `data/accounts.json`.(настроенный аккаунт будет синхронизирован между двумя ботами)
  - Привязка к `telegram_user_id`.
- **Операции от лица пользователя**
  - (В message_fuckerr_bot будет использоваться только для отправки сгенерированных сообщений в личку человека в ответ на его сообщения из группы, которые были найдены скриптом класификатором)
  -(В warp_chat_bot будет использоваться в полном обьеме, чтение, отправка, удаление, редактирование, ответ на личные сообщения, для чего он собственно и был создан)
  - В message_fuckerr_bot по требованию поднимать `TelegramClient(StringSession(...), API_ID, API_HASH)` и выполнить действия (читать/отвечать).
  - В warp_chat_bot держать фоновое подключение на пользователя (отдельная асинхронная таска).
- **Отзыв**
  - `/unlink` удаляет запись и, по возможности, `log_out()` у Telethon.

### Нефункциональные требования
- **Конфиг** ('config.py'):
  - `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` (обязательные; строка 60 в `message_finder.py` уже проверяет).
  - `ACCOUNTS_FILE` (путь к JSON, где будут храниться сесии аккаунтов).

- **Устойчивость**
  - Обработать `FloodWait`, `SessionPasswordNeededError`, `PhoneCodeInvalidError`, `PhoneCodeExpiredError`, DC migration.
  - Ретраи с бэкоффом (можно переиспользовать `utils/retry.py`).
- **Совместимость с текущим скриптом**
  -Код между двумя ботами должен быть максимально общим, и переиспользуемым чтобы не дублировать код и быть дной системой.

### Структура данных (JSON)
```json
{
  "users": [
    {
      "telegram_user_id": 123456789,
      "phone_e164_masked": "+380********",
      "string_session": "string_session_value",
      "status": "active",
      "created_at": "2025-09-17T12:34:56Z",
      "updated_at": "2025-09-17T12:34:56Z",
      "meta": { "method": "phone|qr", "last_ip": null }
    }
  ]
}
```

### Изменения по проекту (файлы)
- `config.py`: добавить `ACCOUNTS_FILE`, убедиться, что `TELEGRAM_API_ID/HASH` обязательны.
- `const.py`: константы команд и UI-состояний: `ACCOUNT_AWAITING_PHONE`, `ACCOUNT_AWAITING_CODE`, `ACCOUNT_AWAITING_2FA`, `ACCOUNT_QR_WAIT`, `ACCOUNT_IDLE`, тексты подсказок.
- `tg/ui_state.py`: добавить новые состояния диалога `/account`.
- `utilities/accounts_store.py` (новый): 
  - `get_user_account(telegram_user_id) -> Optional[Account]`
  - `save_or_update_account(account: Account) -> None`
  - `delete_account(telegram_user_id) -> None`
  - Шифрование/дешифрование `StringSession`.
- `tg/worker.py` или новый модуль `tg/account.py`:
  - Хендлеры `/account`, `/account status`, `/unlink`, `/account help`.
  - Машина состояний и валидация ввода.
  - Телеграм-логин через Telethon: phone+code(+2FA) и QR.
- `services/` (опционально `services/user_sessions.py`):
  - Обёртки над Telethon для создания/восстановления клиента, логина, QR, логаута.
- `logging_config.py`: настройки для маскирования PII.

### Алгоритмы (вкратце)
- **/account (номер)**
  1) Проверить, есть ли активная сессия → если есть, предложить `/account status` или `/unlink`.
  2) Запросить номер → `send_code_request`.
  3) Запросить код → `sign_in(phone, code)`; если требуется 2FA → запросить пароль → `sign_in(password=...)`.
  4) `session = client.session.save()` → шифровать → сохранить.
  5) Ответить успехом, показать `/unlink` и `/account status`.
- **/account (QR)**
  1) `qr_login = client.qr_login()` → получить изображение/ASCII → отправить пользователю.
  2) Дождаться `await qr_login.wait()`.
  3) Сохранить `StringSession`, как выше.

### Важные edge cases
- Включён 2FA → требовать пароль.
- Срок кода истёк → повторная отправка, ограничить число попыток.
- FloodWait → уведомить, поставить на бэкофф/повтор.
- Пользователь отменил → очистить временное состояние.

### Тест-кейсы (акцептанс)
- `/account` → phone → code → success → `status` показывает active.
- `/account` при наличии active → предлагает `status/unlink`.
- `/account` → неверный код → корректное сообщение об ошибке, повтор.
- `/account` → 2FA → ввод пароля → success.
- `/account` → QR → success.
- `/unlink` → удаляет запись, повторный `status` → no account.
- Поднятие клиента из сохранённой сессии и отправка сообщения в выбранный диалог при генерации ответа.(для message_fuckerr_bot)
- Параллельно основной клиент продолжает слушать группы (нет регрессий).

### Переменные окружения
- `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` (обязательные).
- `ACCOUNTS_FILE` (по умолчанию `data/accounts.json`).

### Зависимости
- `telethon`

### Короткие сниппеты

- Генерация/сохранение сессии:
```python
from telethon import TelegramClient
from telethon.sessions import StringSession

async def login_by_code(phone: str, api_id: int, api_hash: str) -> str:
    client = TelegramClient(StringSession(), api_id, api_hash)
    await client.connect()
    sent = await client.send_code_request(phone)
    # далее принять код/2FA паролем
    # await client.sign_in(phone=phone, code=code)
    # или await client.sign_in(password=password)
    s = client.session.save()
    await client.disconnect()
    return s
```

- Восстановление клиента по сохранённой сессии:
```python
def client_from_session(session_str: str, api_id: int, api_hash: str) -> TelegramClient:
    return TelegramClient(StringSession(session_str), api_id, api_hash)
```

### Что отдать агенту-кодеру
- Этот список требований.
- Доступ к коду, env с `TELEGRAM_API_ID/HASH`.
- Ожидаемый путь хранения `ACCOUNTS_FILE`.
- Приоритет: сначала онбординг и сохранение сессии; затем `/unlink` и `status`; затем фоновое прослушивание (если нужно).

### Архитектура и деплой

- Один репозиторий и один Docker-образ `bot_orchestrator`.
- Два процесса (контейнера) в `docker-compose`:
  - message_finder:
    - Слушает группы через Telethon (единый `API_ID/API_HASH`).
    - Классифицирует сообщения и отправляет уведомления пользователям через `message_fuckerr_bot`.
    - Не хранит и не поднимает пользовательские сессии постоянно.
  - warp_chat:
    - Бот-терминал аккаунтов (`/account`, `/unlink`).
    - Онбординг MTProto сессий (номер+код+2FA и QR).
    - Держит фоновые подключения per-user (по необходимости) и выполняет операции (читать/отвечать/удалять/редактировать).
- Общий стор сессий: `ACCOUNTS_FILE=/app/data/accounts.json` (общий volume `./data:/app/data`) — синхронизирует аккаунты между двумя ботами.
- Разделение ответственности:
  - Общий код в `services/user_sessions.py`, `utilities/accounts_store.py`, `tg/account.py` (FSM), `tg/ui_state.py`, `tg/ui.py`.
  - Бот-токены изолированы: message_finder использует `MESSAGE_FUCKERR_TOKEN`, warp_chat — `WARP_CHAT_BOT_TOKEN`.
- Масштабирование:
  - Можно масштабировать независимыми репликами `message_finder` без риска двойной отправки (держать дедупликацию на уровне стора/кворума).
  - `warp_chat` обычно в 1 экземпляре (из-за Bot API long-poll/webhook), при необходимости — перенос на webhook+балансировку.
