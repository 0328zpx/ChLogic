"""Shared utilities for ChLogic evaluation scripts."""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

VALID_EN_LABELS = {"yes", "no", "unknown"}
VALID_ZH_LABELS = {"是", "否", "未知"}


def read_csv(path: Path, required_columns: Sequence[str]) -> List[Dict[str, Any]]:
    """Read a UTF-8 CSV file and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = set(required_columns) - fieldnames
        if missing:
            raise ValueError(
                f"Dataset {path} is missing required columns: {sorted(missing)}"
            )
        return list(reader)


def normalize_label_en(text: object) -> Optional[str]:
    """Normalize English, Chinese, and common label variants."""
    if text is None:
        return None
    raw = str(text).strip().lower()
    if not raw:
        return None

    cleaned = re.sub(
        r"[\s\.,!?:;\"'`()\[\]{}，。！？：；、“”‘’]+", " ", raw
    ).strip()

    direct = {
        "yes": "yes", "y": "yes", "true": "yes", "correct": "yes",
        "valid": "yes", "entailed": "yes", "entails": "yes",
        "是": "yes", "真": "yes", "正确": "yes", "成立": "yes",
        "no": "no", "n": "no", "false": "no", "incorrect": "no",
        "invalid": "no", "not valid": "no", "not entailed": "no",
        "does not follow": "no", "否": "no", "假": "no",
        "错误": "no", "不成立": "no",
        "unknown": "unknown", "unknow": "unknown", "uncertain": "unknown",
        "undetermined": "unknown", "cannot determine": "unknown",
        "cannot be determined": "unknown",
        "not enough information": "unknown",
        "insufficient information": "unknown",
        "not necessarily": "unknown", "未知": "unknown",
        "不确定": "unknown", "无法确定": "unknown",
        "不能确定": "unknown", "信息不足": "unknown",
    }
    if cleaned in direct:
        return direct[cleaned]

    patterns = [
        ("unknown", [
            r"\bunknown\b", r"\bunknow\b", r"\buncertain\b",
            r"\bundetermined\b", r"\bcannot\s+(be\s+)?determined\b",
            r"\bnot\s+enough\s+information\b",
            r"\binsufficient\s+information\b", r"\bnot\s+necessarily\b",
            r"未知|不确定|无法确定|不能确定|信息不足|无法判断|不能判断",
        ]),
        ("no", [
            r"\bno\b", r"\bfalse\b", r"\bincorrect\b", r"\binvalid\b",
            r"\bdoes\s+not\s+follow\b", r"\bnot\s+entailed\b",
            r"\bnot\s+valid\b", r"否|错误|不成立|不能推出|无法推出|推不出",
        ]),
        ("yes", [
            r"\byes\b", r"\btrue\b", r"\bcorrect\b", r"\bvalid\b",
            r"\bentailed\b", r"\bentails\b", r"是|正确|成立|能推出|可以推出|必然推出",
        ]),
    ]
    for label, label_patterns in patterns:
        if any(re.search(pattern, cleaned) for pattern in label_patterns):
            return label
    return None


def normalize_label_zh(text: object) -> Optional[str]:
    normalized = normalize_label_en(text)
    return {"yes": "是", "no": "否", "unknown": "未知"}.get(normalized)


def init_stat() -> Dict[str, int]:
    return {"total": 0, "correct": 0, "invalid_pred": 0}


def update_stat(stat: Dict[str, int], is_correct: bool, is_valid: bool) -> None:
    stat["total"] += 1
    stat["correct"] += int(is_correct)
    stat["invalid_pred"] += int(not is_valid)


def accuracy(stat: Dict[str, int]) -> float:
    return stat["correct"] / stat["total"] if stat["total"] else 0.0


def call_with_retry(callable_fn, max_retries: int, wait_seconds: float) -> str:
    """Call a zero-argument function with bounded retries."""
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return callable_fn()
        except Exception as exc:  # API/network errors vary by provider.
            last_error = exc
            print(f"[WARN] API call {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(wait_seconds)
    assert last_error is not None
    raise last_error


def write_results(
    results: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str,
    summary: Dict[str, Any],
) -> None:
    """Write item-level JSONL/CSV and aggregate JSON outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{prefix}_predictions.jsonl"
    csv_path = output_dir / f"{prefix}_predictions.csv"
    summary_path = output_dir / f"{prefix}_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    if results:
        fieldnames = list(results[0].keys())
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote {jsonl_path}")
    if results:
        print(f"[INFO] Wrote {csv_path}")
    print(f"[INFO] Wrote {summary_path}")
