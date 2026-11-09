"""JSON input validation and linguistic-density window rules."""
import hashlib
import json
import math
import re


def load_rows(path, limit=None):
    with open(path, encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f'{path}: expected nonempty object mapping labels to text lists')
    rows = []
    for label, texts in obj.items():
        if not isinstance(texts, list):
            raise ValueError(f'{path}: {label} must map to a list')
        for i, text in enumerate(texts[:limit]):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'{path}: invalid text at {label}[{i}]')
            digest = hashlib.sha1(text.encode()).hexdigest()[:16]
            rows.append(dict(sample_id=f'{label}::{i}::{digest}', label=str(label), text=text))
    if not rows:
        raise ValueError(f'{path}: no samples')
    return rows


def effective_length(text):
    n = len(text.split())
    sentences = max(1, len([s for s in re.split(r'[.!?]+', text) if s.strip()]))
    punctuation = len(re.findall(r'[,;:—\-(){}\[\]…。、；：]', text))
    return n / sentences * (1 - min(.9, punctuation / max(n, 1)))


def get_text_k_metrics(text: str) -> float:
    """Compute the per-text window using Equation (1)."""
    if not text.split():
        return 2.0
    return float(max(2, math.floor(math.sqrt(effective_length(text)))))


def dataset_window(rows):
    """Average per-text windows on training texts and round up for a shared window."""
    if not rows:
        raise ValueError("Cannot calculate a window from an empty training dataset.")
    return max(2, math.ceil(sum(get_text_k_metrics(r['text']) for r in rows) / len(rows)))
