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
    for ex in examples_data:
        message = ex.get("message", "")
        output = ex.get("output", {})
        label = ex.get("label")

        if not message:
            continue

        # 1) Пытаемся взять классификацию из output.classification или строкового output
        classification_value: str = ""
        if isinstance(output, dict):
            try:
                classification_value = str(output.get("classification", "")).strip()
            except Exception:
                classification_value = ""
        elif isinstance(output, str):
            ov = output.strip()
            if ov in {"0", "1"}:
                classification_value = ov

        # 2) Если в output нет валидного значения, используем label (0/1)
        if classification_value not in {"0", "1"}:
            if label in (0, 1):
                classification_value = "1" if label == 1 else "0"
            else:
                # пропускаем записи без корректной разметки
                continue

        # Для нового бинарного формата обучающих примеров возвращаем только 0/1 без JSON
        formatted_examples.append(
            f'Input: {json.dumps(message, ensure_ascii=False)}\nOutput: {classification_value}'
        )

    return formatted_examples


