
#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible chat model on a Chinese ChLogic column."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from common import (
    VALID_ZH_LABELS,
    accuracy,
    call_with_retry,
    init_stat,
    normalize_label_en,
    normalize_label_zh,
    read_csv,
    update_stat,
    write_results,
)


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=env("CHLOGIC_DATASET_PATH", "data/general_aligned.csv"),
        help="CSV dataset path (env: CHLOGIC_DATASET_PATH).",
    )
    parser.add_argument(
        "--input-column",
        default=env("CHLOGIC_INPUT_COLUMN", "zh_rhetorical_implicit"),
        help="Chinese text column (env: CHLOGIC_INPUT_COLUMN).",
    )
    parser.add_argument(
        "--label-column",
        default=env("CHLOGIC_LABEL_COLUMN", "label_zh"),
        help="Gold-label column (env: CHLOGIC_LABEL_COLUMN).",
    )
    parser.add_argument(
        "--logic-type-column",
        default=env("CHLOGIC_LOGIC_TYPE_COLUMN", "logic_type"),
    )
    parser.add_argument("--model", default=env("CHLOGIC_MODEL_NAME"))
    parser.add_argument("--base-url", default=env("CHLOGIC_BASE_URL"))
    parser.add_argument("--api-key-env", default="CHLOGIC_API_KEY")
    parser.add_argument("--output-dir", default=env("CHLOGIC_OUTPUT_DIR", "outputs"))
    parser.add_argument(
        "--output-prefix",
        default=env("CHLOGIC_OUTPUT_PREFIX", "chinese_evaluation"),
    )
    parser.add_argument("--temperature", type=float, default=float(env("CHLOGIC_TEMPERATURE", "0")))
    parser.add_argument("--max-tokens", type=int, default=int(env("CHLOGIC_MAX_TOKENS", "16")))
    parser.add_argument("--max-retries", type=int, default=int(env("CHLOGIC_MAX_RETRIES", "3")))
    parser.add_argument("--retry-wait", type=float, default=float(env("CHLOGIC_RETRY_WAIT_SECONDS", "3")))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=env("CHLOGIC_ENABLE_THINKING", "false").lower() == "true",
    )
    return parser


def build_prompt(query_text: str) -> str:
    return f"""你现在要完成一个中文逻辑推理判断任务。

请仔细阅读下面的事实和问题，判断问题中的结论是否能由事实必然推出。

你只能输出下面三个选项中的一个：
是
否
未知

含义：
- 是：结论能由事实必然推出。
- 否：结论为假，或者该推理形式无效。
- 未知：事实信息不足，不能确定结论是否成立。

不要输出解释、标点或其他内容。

题目：
{query_text}"""


# 新增：General / Difficult 专用二分类提示词
def build_prompt_binary(query_text: str) -> str:
    return f"""你现在要完成一个中文逻辑推理判断任务。

请仔细阅读下面的事实和问题，判断问题中的结论是否能由事实必然推出。

你只能输出下面两个选项中的一个：
是
否

含义：
- 是：结论能由事实必然推出。
- 否：结论不能由事实必然推出。

不要输出解释、标点或其他内容。

题目：
{query_text}"""


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set environment variable {args.api_key_env}."
        )
    if not args.model:
        raise RuntimeError("Missing model name. Set CHLOGIC_MODEL_NAME or use --model.")
    if not args.base_url:
        raise RuntimeError("Missing API endpoint. Set CHLOGIC_BASE_URL or use --base-url.")

    dataset_path = Path(args.dataset)
    rows = read_csv(
        dataset_path,
        [args.logic_type_column, args.input_column, args.label_column],
    )
    if args.limit is not None:
        rows = rows[: args.limit]

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    overall = init_stat()
    by_logic_type: Dict[str, Dict[str, int]] = {}
    by_label: Dict[str, Dict[str, int]] = {}
    results = []

    for index, row in enumerate(rows, start=1):
        logic_type = str(row.get(args.logic_type_column, "")).strip()
        query = str(row.get(args.input_column, "")).strip()
        gold_raw = str(row.get(args.label_column, "")).strip()
        gold = normalize_label_zh(gold_raw)

        if not query:
            print(f"[WARN] Row {index}: empty input; skipped.")
            continue
        if gold not in VALID_ZH_LABELS:
            print(f"[WARN] Row {index}: invalid gold label {gold_raw!r}; skipped.")
            continue

        # 修改：根据数据集名称选择二分类或三分类提示词
        if dataset_path.name in {"general_aligned.csv", "difficult_aligned.csv"}:
            prompt = build_prompt_binary(query)
        else:
            prompt = build_prompt(query)

        def api_call() -> str:
            kwargs: Dict[str, Any] = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "extra_body": {"enable_thinking": bool(args.enable_thinking)},
            }
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""

        raw_output = call_with_retry(api_call, args.max_retries, args.retry_wait)
        prediction = normalize_label_zh(raw_output)
        is_valid = prediction in VALID_ZH_LABELS
        is_correct = bool(is_valid and prediction == gold)

        update_stat(overall, is_correct, is_valid)
        by_logic_type.setdefault(logic_type, init_stat())
        update_stat(by_logic_type[logic_type], is_correct, is_valid)
        by_label.setdefault(gold, init_stat())
        update_stat(by_label[gold], is_correct, is_valid)

        results.append(
            {
                "index": index,
                "logic_type": logic_type,
                "input_column": args.input_column,
                "input_text": query,
                "gold_label_zh_raw": gold_raw,
                "gold_label_zh": gold,
                "gold_label_en": normalize_label_en(gold),
                "raw_output": raw_output,
                "pred_label_zh": prediction,
                "pred_label_en": normalize_label_en(prediction),
                "is_valid_prediction": is_valid,
                "is_correct": is_correct,
            }
        )

        if index % 20 == 0:
            print(
                f"[INFO] Completed {index} rows | "
                f"accuracy={accuracy(overall):.6f} | "
                f"invalid={overall['invalid_pred']}"
            )

    summary = {
        "model_name": args.model,
        "base_url": args.base_url,
        "dataset_path": str(dataset_path),
        "input_column": args.input_column,
        "label_column": args.label_column,
        "total": overall["total"],
        "correct": overall["correct"],
        "invalid_predictions": overall["invalid_pred"],
        "overall_accuracy": accuracy(overall),
        "logic_type_accuracies": {
            key: {**value, "accuracy": accuracy(value)}
            for key, value in sorted(by_logic_type.items())
        },
        "label_accuracies": {
            key: {**value, "accuracy": accuracy(value)}
            for key, value in sorted(by_label.items())
        },
    }
    write_results(results, Path(args.output_dir), args.output_prefix, summary)


if __name__ == "__main__":
    main()
