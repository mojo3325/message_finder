## TDD — Технический дизайн: Скрипт-классификатор сообщений Telegram

### 1. Обзор архитектуры
- **Процесс**: один долгоживущий воркер внутри Docker-контейнера.
- **Основные модули**:
  - `telegram_listener`: подписка на новые сообщения только из групп/супергрупп/каналов (без личных диалогов) от имени аккаунта.
  - `classifier`: обёртка над официальным Python SDK Cerebras для вызова модели `qwen-3-235b-a22b-instruct-2507` со строгим системным промптом и валидацией ответа `0/1`.
  - `subscriber_store`: хранение `user_id`, запустивших бота `/start` (in-memory + файл снапшота/volume).
  - `bot_notifier`: отправка ЛС через Bot API только пользователям из `subscriber_store`.
  - `bot_ui`: inline-клавиатура (Generate/Back/Regenerate), редактирование сообщений, in-memory `reply_ui_store`.
  - `rate_limiter`: ограничение скорости запросов к Cerebras (token bucket для RPM/RPH и контроль TPM).
  - `dedup_store`: идемпотентность по message_id (например, in-memory + optional файл/БД в будущем).
  - `logging_metrics`: структурированные логи и метрики.

### 2. Потоки данных
1) `telegram_listener` получает апдейт `new_message` только из общих чатов с полями: `chat_id`, `message_id`, `from_user`, `text`, `chat_meta`.
2) Проверка дубликата в `dedup_store`. Если уже обработан — пропуск.
3) Если сообщение содержит чистый текст (без медиа/голосовых), через `rate_limiter` пропускается вызов `classifier.classify(text)` → запрос к Cerebras, ответ строго `"0"` или `"1"`. Иначе — пропуск.
4) Если `"1"` — попытка определить `author_user_id`. Если автор определён и `subscriber_store.contains(author_user_id)` — `bot_notifier.send_personal(author_user_id, payload)`. Иначе — пропуск.
5) Запись результата и метрик.

### 3. Интеграции
- **Telegram MTProto (user)**: Python-клиент MTProto (Telethon/Pyrogram). Требуются `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, сессионная строка (StringSession, предпочтительно) или файл сессии.
- **Telegram Bot API**: HTTP-клиент (requests/httpx) или библиотека (aiogram/pyTelegramBotAPI). Требуется `TELEGRAM_BOT_TOKEN`. Поддерживаются команды `/start` (подписка) и `/stop` (отписка).
- **Cerebras**: официальный Python SDK. Требуется `CEREBRAS_API_KEY`. Модель: `qwen-3-235b-a22b-instruct-2507`.

### 4. Строгий системный промпт
- Требование: вернуть только `0` или `1` без пробелов/комментариев/форматирования.
- Черновик промпта:
```
System: You are a strict binary classifier for "newbie questions" in chat messages. Output must be exactly a single character: 0 or 1. Never explain.
Definition: Output 1 if the message is a newbie-style question seeking basic guidance (e.g., how to withdraw crypto to a card in Russia, how to connect a wallet to HyperLiquid via OKX, how to start, what to click, where to find, etc.). Otherwise output 0.
Rules: Return only 0 or 1. No spaces. No punctuation. No words. No JSON. No code.
```

### 5. Вызов SDK Cerebras (псевдокод)
```python
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])  # имя инициализации может отличаться по документации

SYSTEM_PROMPT = (
    "You are a strict binary classifier for \"newbie questions\" in chat messages. "
    "Output must be exactly a single character: 0 or 1. Never explain. "
    "Definition: Output 1 if the message is a newbie-style question seeking basic guidance. "
    "Rules: Return only 0 or 1. No spaces. No punctuation. No words. No JSON. No code."
)

def classify_with_cerebras(message_text: str) -> str:
    if not message_text or not message_text.strip():
        return "0"

    # rate limit: ожидание токена перед вызовом
    rate_limiter.acquire()

    response = client.chat.completions.create(
        model="qwen-3-235b-a22b-instruct-2507",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message_text.strip()},
        ],
        temperature=0.0,
        max_tokens=1,
    )
    raw = response.choices[0].message.content.strip()
    return "1" if raw == "1" else "0"
```

Примечание: Уточнить точные классы/методы SDK по официальной документации Cerebras и обновить имена, если отличаются.

### 6. Telegram слушатель (псевдокод на Telethon)
```python
from telethon import TelegramClient, events
from telethon.sessions import StringSession

api_id = int(os.environ["TELEGRAM_API_ID"])
api_hash = os.environ["TELEGRAM_API_HASH"]
string_session = os.environ.get("TELEGRAM_STRING_SESSION")
if string_session:
    client = TelegramClient(StringSession(string_session), api_id=api_id, api_hash=api_hash)
else:
    # fallback: файловая сессия (требуется примонтированный volume)
    session_path = os.environ.get("TELEGRAM_SESSION_PATH", "session")
    client = TelegramClient(session=session_path, api_id=api_id, api_hash=api_hash)

@client.on(events.NewMessage(chats=None))  # слушать все, но фильтровать тип чата внутри
async def handler(event):
    # 1) исключаем личные диалоги
    if event.is_private:
        return

    # 2) обрабатываем только текст
    if not event.message or not event.message.message:
        return
    text = event.message.message
    from_user = await event.get_sender()
    chat = await event.get_chat()

    # идемпотентность
    if dedup_store.seen(event.chat_id, event.id):
        return

    # 3) классификация только для текстовых
    label = classify_with_cerebras(text)
    if label == "1":
        link = await build_message_link(event)
        # Отправка через бота только пользователям, запустившим /start
        if subscriber_store.contains(from_user.id):
            await bot_notifier_send(from_user.id, chat, text, link)

    dedup_store.mark(event.chat_id, event.id)

client.start()
client.run_until_disconnected()
```

### 7. Формирование ссылки на сообщение
- Публичные чаты/каналы с username: `https://t.me/<username>/<message_id>`.
- Приватные/супергруппы без username: `https://t.me/c/<internal_chat_id>/<message_id>` (internal_chat_id = abs(chat_id) без -100 префикса; завист от клиента/контекста).
- Обрабатывать ошибки и недоступность ссылки (fallback: без ссылки).

### 8. Формат уведомления (Bot API)
- Текст ЛС (MarkdownV2 или HTML):
  - Имя: `<Имя Фамилия | отсутствует>`
  - Ник: `@username | отсутствует`
  - Чат: `<title | type>`
  - Сообщение: подчеркнуто (например, HTML `<u>текст</u>`)
  - Ссылка: кликабельная, если возможно
  - Ограничение: бот может написать только пользователю, который запустил `/start`.
  - Отписка: команда `/stop` удаляет пользователя из `subscriber_store` и прекращает дальнейшие уведомления.
  - Кнопки UI:
    - Стартовая: `⚡ Сгенерировать ответ` → генерирует через Cerebras ответ (1–3 фразы, дружелюбно, вовлекающе), заменяет тело сообщения; кнопки меняются на `⬅ Назад` и `🔁 Перегенерировать`.
    - `📋 Скопировать` добавляется как `copy_text` при поддержке клиента Telegram; при ошибке Bot API выполняется фолбэк без этой кнопки.

### 9. Идемпотентность и хранение состояния
- In-memory `LRU`/`TTL` кэш для `(chat_id, message_id)`.
- Опционально: файл-снапшот для восстановления после рестарта.
 - `reply_ui_store`: in-memory map `sid -> {user_id, original_html, original_text, last_reply}`; используется для `gen/back/regen`.

### 10. Повторные попытки и таймауты
- SDK Cerebras: таймаут `REQUEST_TIMEOUT_S` (env), retry с экспоненциальной задержкой до `RETRY_MAX_ATTEMPTS`.
- Telegram отправка уведомления: retry по тем же правилам.
 - Rate limit: token bucket по `RATE_LIMIT_RPM`/`RATE_LIMIT_RPH` и контроль `RATE_LIMIT_TPM`. При 429 — читать `Retry-After`, ставить задачу в очередь с отложенным повтором.

### 11. Конфигурация (ENV)
- `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_STRING_SESSION` (предпочтительно) или `TELEGRAM_SESSION_PATH`
- `TELEGRAM_BOT_TOKEN`
- `CEREBRAS_API_KEY`
 - `RATE_LIMIT_RPM` (default: 30)
 - `RATE_LIMIT_RPH` (default: 900)
 - `RATE_LIMIT_TPM` (default: 60000)
- `LOG_LEVEL` (default: INFO)
- `REQUEST_TIMEOUT_S` (default: 20)
- `RETRY_MAX_ATTEMPTS` (default: 3)

### 12. Docker
- Базовый образ: `python:3.11-slim`.
- Установка зависимостей: Telethon/Pyrogram, официальный Cerebras SDK, httpx/requests, uvloop (опц.).
- Контейнер запускает один процесс-воркер.

#### 12.1 Персистентная сессия (без повторного ввода кода/облачного пароля)
- Сгенерировать один раз StringSession локально (вне контейнера) через Telethon скрипт авторизации (ввод кода/пароля делается один раз).
- Сохранить строку в секрет-хранилище/ENV `TELEGRAM_STRING_SESSION`.
- Либо использовать файловую `.session` на примонтированном volume и задавать `TELEGRAM_SESSION_PATH`.

### 12.2 Генерация ответа (Cerebras)
- System prompt для генерации:
  - Короткие (1–3 предложения), дружелюбные, языконейтральные, без markdown/ссылок/дисклеймеров.
- Температура: `0.7`, `max_tokens: 160`.

### 13. Тестирование
- **Модульные**: 
  - парсинг Telegram событий; 
  - нормализация ответа Cerebras; 
  - генерация ссылок.
- **Интеграционные**: мок SDK Cerebras (температура=0, max_tokens=1), мок Telegram клиента.
- **Нагрузочные**: генерация N сообщений/мин, p95 latency ≤ 1.5s.
- **Надёжность**: симуляция ошибок сети/5xx, проверка retry и идемпотентности.

### 14. Логи и метрики
- Логи JSON: уровень, событие, chat_id, message_id, latency, label, ошибки.
- Метрики (экспорт в stdout или Prom endpoint в будущем):
  - `messages_total`, `classified_total`, `label_one_total`, `errors_total`, `latency_ms_histogram`.

### 15. Безопасность
- Ключи и сессии только через ENV/секреты, без логирования значений.
- Минимизировать права доступа аккаунта.

### 16. Ограничения V1 и будущие улучшения
- V1: текст только; нет персистентного стора; один аккаунт.
- V2+: Redis для идемпотентности, Prometheus, several workers, мультиязычные эвристики, контент-модерация.
