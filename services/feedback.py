import json
import os
from typing import Any, Iterable, List, Mapping, Optional

import filelock


FEEDBACK_FILE_PATH = "data/feedback.json"
LOCK_FILE_PATH = "data/feedback.json.lock"


def save_feedback(feedback_data: dict) -> None:
    os.makedirs(os.path.dirname(FEEDBACK_FILE_PATH), exist_ok=True)
    lock = filelock.FileLock(LOCK_FILE_PATH)
    with lock:
        examples: list = []
        if os.path.exists(FEEDBACK_FILE_PATH):
            try:
                with open(FEEDBACK_FILE_PATH, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        examples = json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError):
                examples = []

        examples.append(feedback_data)

        with open(FEEDBACK_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)


def _coerce_examples(data: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(data, Mapping):
        yield data
        return

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, Mapping):
                yield entry


def load_feedback_examples() -> List[str]:
    if not os.path.exists(FEEDBACK_FILE_PATH):
        return []
    lock = filelock.FileLock(LOCK_FILE_PATH)
    with lock:
        try:
            with open(FEEDBACK_FILE_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    return []
                examples_data = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    formatted_examples: List[str] = []
    for i, ex in enumerate(_coerce_examples(examples_data)):
        message_raw = ex.get("message")
        if isinstance(message_raw, bytes):
            try:
                message = message_raw.decode("utf-8", errors="ignore").strip()
            except Exception:  # noqa: BLE001
                message = ""
        elif isinstance(message_raw, str):
            message = message_raw.strip()
        elif message_raw is not None:
            message = str(message_raw).strip()
        else:
            message = ""

        output_raw = ex.get("output")
        if isinstance(output_raw, bytes):
            try:
                output_clean = output_raw.decode("utf-8", errors="ignore").strip()
            except Exception:  # noqa: BLE001
                output_clean = ""
        elif isinstance(output_raw, str):
            output_clean = output_raw.strip()
        elif isinstance(output_raw, (int, float)):
            output_clean = str(int(output_raw)).strip()
        else:
            output_clean = ""

        label_raw = ex.get("label")
        label_clean: Optional[int]
        if isinstance(label_raw, int) and label_raw in (0, 1):
            label_clean = label_raw
        elif isinstance(label_raw, str) and label_raw.strip() in {"0", "1"}:
            label_clean = int(label_raw.strip())
        else:
            label_clean = None

        if not message:
            continue

        # Определяем правильный label для классификации
        classification_label: Optional[int] = None
        if output_clean in {"0", "1"}:
            classification_label = int(output_clean)
        elif label_clean is not None:
            classification_label = label_clean

        if classification_label is None:
            # Пропускаем записи без корректной разметки
            continue

        # Форматируем в стиле примеров из CLASSIFIER_PROMPT
        input_json = json.dumps({"items": [{"id": str(i + 1), "text": message}]}, ensure_ascii=False)
        output_json = json.dumps({"items": [{"id": str(i + 1), "label": classification_label}]}, ensure_ascii=False)
        
        formatted_examples.append(f"Input:\n{input_json}\nOutput:\n{output_json}")

    return formatted_examples


