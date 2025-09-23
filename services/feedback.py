import json
import os
from typing import List

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
    for i, ex in enumerate(examples_data):
        message = ex.get("message", "")
        output = ex.get("output", "")
        label = ex.get("label")

        if not message:
            continue

        # Определяем правильный label для классификации
        classification_label: int = 0
        if isinstance(output, str) and output.strip() in {"0", "1"}:
            classification_label = int(output.strip())
        elif isinstance(label, int) and label in (0, 1):
            classification_label = label
        else:
            # Пропускаем записи без корректной разметки
            continue

        # Форматируем в стиле примеров из CLASSIFIER_PROMPT
        input_json = json.dumps({"items": [{"id": str(i + 1), "text": message}]}, ensure_ascii=False)
        output_json = json.dumps({"items": [{"id": str(i + 1), "label": classification_label}]}, ensure_ascii=False)
        
        formatted_examples.append(f"Input:\n{input_json}\nOutput:\n{output_json}")

    return formatted_examples


