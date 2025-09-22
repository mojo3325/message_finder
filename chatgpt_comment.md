Коротко: у тебя не «магия сломалась», у тебя бесплатные лимиты душат трафик, плюс код сам себя душит повторными ретраями. «429» и «Tokens per day limit exceeded» из логов — это не тонкий художественный намёк, это прямой приказ тормозить.

Что с лимитами Mistral и моделью
	•	Модель существует и именно так и зовётся: magistral-small-2509. Сейчас алиас magistral-small-latest указывает на неё. Так что имя ок, а переключаться безопаснее на *-latest.  ￼
	•	У Mistral есть бесплатный API-тариф, но точных цифр публично не пишут: лимиты смотрятся в консоли «La Plateforme». Типы ограничений: RPS, токены в минуту и в месяц на организацию. Да, бесплатно «скромно».  ￼

Побочный трэш в логах:
	•	cerebras_tpd_switch_key и «Tokens per day limit exceeded» — это уже твой запасной провайдер Cerebras во free-режиме уткнулся в суточную квоту. Никакой магии, просто потолок.
	•	«mistral_retry … 429 rate» — классическая перегрузка по RPS/TPM на их стороне.

Почему сейчас больно
	1.	Двойные ретраи. У тебя и SDK ретраит, и твой код ретраит — получается «штурм в лоб» и лавина 429.
	2.	Мелкоточечные вызовы «по одному сообщению» растят RPS.
	3.	Вывод болтливый: без жёсткого JSON-режима модель легко тратит лишние токены.
	4.	Нет адаптивного троттлинга по реальным ответам сервера.

Быстрые фиксы (с минимумом страданий)
	1.	JSON-режим + короткий ответ. Попроси строго JSON и зафиксируй микросхему. На стороне Mistral это response_format={"type":"json_object"} или кастомная схема. Темп 0, max_output_tokens 16–32.  ￼
	2.	Микробатчинг. Копи 800 мс входящих сообщений в пачку и классифицируй списком за один вызов. RPS падает в разы, TPM почти не растёт.
	3.	Адаптивный троттлинг. При 429:
	•	уважай Retry-After если прислали;
	•	увеличивай глобальный интервал между запросами (например, на 25–50%) на 1–3 минуты;
	•	снижай параллелизм до 1.
	4.	Единственный слой ретраев. Отключи ретраи SDK или свои. Оставь один механизм с экспоненциальной задержкой и джиттером.
	5.	Микро-промпт. Убирай всё лишнее из system/user. Классификатору не нужна поэзия.
	6.	Хитрый оффлоад. Тривиальные сообщения (команды вида !c blum, голые цены, «Согласен», «Ну норм») разбирай правилами/регексами локально. На LLM отправляй только серую зону. Это легко срежет 50% обращений.
	7.	Batch API для бэклога. Если есть хвост непросрочных сообщений, скидывай в batch-инференс вместо онлайна.  ￼
	8.	Пределы на день по каждому провайдеру. Держи счётчик своих токенов, чтобы не влетать в TPD у Cerebras. На пороге — авто-свайч или graceful-дефолт.

Точные правки по коду

Ниже минимальный «патч-скелет», который устраняет главные проблемы. Подстрой под свои функции.

1) Вызов Mistral с жёстким JSON и коротким ответом

from mistralai import Mistral

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def classify_batch(messages: list[dict]) -> list[dict]:
    # messages = [{"id": "...", "text": "...", "context": "..."}]
    prompt = {
        "role": "user",
        "content": (
            "Ты — классификатор. Верни JSON со списком объектов {id, label}.\n"
            "Допустимые label: 0,1.\n"
            "Ответ ТОЛЬКО JSON без текста."
            f"\nDATA:\n{messages}"
        ),
    }
    res = client.chat.complete(
        model="magistral-small-latest",
        messages=[prompt],
        temperature=0,
        max_output_tokens=32,
        response_format={"type": "json_object"},
        # при желании: seed=7
    )
    return res.output[0].content[0].text  # парсишь JSON дальше

Док-подтверждение JSON-режима и структурированных ответов здесь.  ￼

2) Микробатчинг 800 мс

from threading import Lock, Event
from collections import deque
import time, json

_queue = deque()
_q_lock = Lock()
_flush_evt = Event()

def enqueue_for_classification(item):
    with _q_lock:
        _queue.append(item)

def _flusher_loop():
    BATCH_MS = 250
    MAX_BATCH = 32
    last = time.monotonic()
    while True:
        time.sleep(0.01)
        now = time.monotonic()
        if (now - last) * 1000 < BATCH_MS:
            continue
        with _q_lock:
            batch = [ _queue.popleft() for _ in range(min(len(_queue), MAX_BATCH)) ]
        if batch:
            try:
                raw = classify_batch(batch)  # см. функцию выше
                # раздать результаты по id...
            except RateLimited as e:
                backoff = e.retry_after or 1.5
                time.sleep(backoff + random.random())
            except Exception:
                # логируй и продолжай, без каскада ретраев
                pass
        last = now

3) Адаптивный троттлинг при 429

GLOBAL_SLEEP = 0.0  # увеличиваем при 429

def mistral_call(*args, **kwargs):
    global GLOBAL_SLEEP
    if GLOBAL_SLEEP:
        time.sleep(GLOBAL_SLEEP)
    try:
        return client.chat.complete(*args, **kwargs)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            ra = float(e.response.headers.get("Retry-After", "1.0"))
            GLOBAL_SLEEP = min(max(GLOBAL_SLEEP * 1.5 + ra, 0.5), 5.0)
            raise RateLimited(retry_after=GLOBAL_SLEEP)
        raise

4) Убери дублирующиеся ретраи
	•	Оставь RETRY_MAX_ATTEMPTS в одном месте. Если SDK уже ретраит, свои ретраи отключи, чтобы не множить запросы при 429.

5) Сократи токены
	•	Считай max_input_tokens на основании твоего estimate_prompt_tokens и руби контекст жестко.
	•	Вывод ограничивай max_output_tokens=16–32. Для бинарной метки тебе не нужно больше.

6) Переход на batch-инференс для «хвостов»
	•	Когда очередь > N, вместо онлайн-кластеризации засылай пачку как batch-job и забирай результат позже. Док здесь.  ￼

Что делать прямо сейчас
	1.	Включи JSON-режим и урежь max_output_tokens.
	2.	Введи микробатчинг и адаптивный троттлинг по 429.
	3.	Отключи второй слой ретраев.
	4.	Для простых кейсов добавь локальные правила и не трать на них LLM.
