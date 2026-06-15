# ChLogic

This repository contains the ChLogic benchmark and evaluation scripts.

## Contents

- `data/general_aligned.csv`: 3,000 General aligned items
- `data/difficult_aligned.csv`: 2,000 Difficult aligned items
- `data/chinese_only.csv`: 1,500 Chinese-specific items
- `code/evaluate_english.py`: evaluation for English inputs
- `code/evaluate_chinese.py`: evaluation for Chinese inputs
- `code/common.py`: shared data-loading, label-normalization, and output utilities

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set the model endpoint, API key, and model identifier as environment variables:

```bash
export CHLOGIC_API_KEY="YOUR_API_KEY"
export CHLOGIC_BASE_URL="https://your-endpoint/v1"
export CHLOGIC_MODEL_NAME="your-model-id"
```

## English evaluation

```bash
python code/evaluate_english.py \
  --dataset data/general_aligned.csv \
  --input-column standard_english \
  --label-column label_en \
  --output-prefix general_english
```

## Chinese evaluation

```bash
python code/evaluate_chinese.py \
  --dataset data/general_aligned.csv \
  --input-column zh_rhetorical_implicit \
  --label-column label_zh \
  --output-prefix general_zh_rhetorical
```

Other Chinese input columns include:

- `zh_literal_translation`
- `zh_natural_written`
- `zh_colloquial`
- `zh_rhetorical_implicit`
- `zh_high_interference`

For the Chinese-only set:

```bash
python code/evaluate_chinese.py \
  --dataset data/chinese_only.csv \
  --input-column full_text \
  --label-column label_zh \
  --output-prefix chinese_only
```

The scripts write item-level predictions and aggregate summaries to the `outputs/` directory by default.
