# Обучение semantic cross-encoder на H100

Инструкция рассчитана на сервер без доступа к Hugging Face Hub и без вызовов
Mistral/OpenRouter. Все пути задаются явно; скрипты не удаляют существующие
файлы.

Если используется готовый архив `airi_mars_training_bundle_v1_20260820.zip`,
выполните инструкцию [TRAINING_BUNDLE.md](TRAINING_BUNDLE.md): в нём корпус,
directional-разметка и ModernBERT уже разложены в ожидаемую структуру.

## 1. Получить код

```bash
cd /home/<user>
git clone https://github.com/KamilIskhakov/AIRI-MARS.git
cd AIRI-MARS
```

## 2. Подготовить Python

Если на сервере уже есть окружение с CUDA-сборкой PyTorch, активируйте его и
установите остальные зависимости:

```bash
python -m pip install -r requirements-training.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Ожидается `torch.cuda.is_available() == True` и NVIDIA H100. Не заменяйте
рабочую CUDA-сборку PyTorch CPU-сборкой. Поэтому `torch` намеренно не включён
в `requirements-training.txt`; при его отсутствии установите подходящую
CUDA-сборку способом, принятым на сервере.

## 3. Разместить внешние артефакты

Рекомендуемая структура может находиться вне Git-репозитория:

```text
/home/<user>/airi_artifacts/
├── corpus/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── ranking_train.jsonl
│   └── manifest.json
├── directional/
│   └── directional_disagreements_v2.jsonl
└── modernbert/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

Источники на рабочей машине:

```text
data/semantic_corpus_v1_20260819/
data/semantic_quality_rejudge_20260819/directional_disagreements_v2.jsonl
models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000/
```

Для базовой модели не нужны `optimizer.pt`, `scheduler.pt`, `rng_state.pth`,
`trainer_state.json` и `training_args.bin`. Обучение создаёт новый optimizer.

Контрольные размеры канонического корпуса:

```text
train.jsonl          19 947 строк
val.jsonl             2 494 строки
test.jsonl            2 494 строки
ranking_train.jsonl  10 975 строк
directional JSONL        38 строк
```

## 4. Обязательный one-step smoke test

```bash
export CORPUS_DIR=/home/<user>/airi_artifacts/corpus
export BASE_MODEL=/home/<user>/airi_artifacts/modernbert
export DIRECTIONAL_JSONL=/home/<user>/airi_artifacts/directional/directional_disagreements_v2.jsonl
export OUTPUT_DIR=/home/<user>/airi_runs/smoke_$(date +%Y%m%d_%H%M%S)

DEVICE=cuda PRECISION=bf16 MAX_LENGTH=128 \
bash mvp/run_multitask_smoke.sh
```

Smoke-runner:

1. Проверяет обязательные файлы и локальную загрузку tokenizer/model.
2. Выбирает по 8 реальных строк train/validation/test с обоими классами.
3. Включает exact-match directional-пример в каждый split.
4. Добавляет одну настоящую ranking-пару.
5. Выполняет один train batch и по одному validation/test проходу.
6. Сохраняет и повторно загружает encoder, tokenizer и четыре головы.

Успех подтверждается строкой `SMOKE TEST PASSED` и файлами:

```text
$OUTPUT_DIR/model/best/multitask_heads.pt
$OUTPUT_DIR/model/best/multitask_config.json
$OUTPUT_DIR/model/metrics.jsonl
$OUTPUT_DIR/model/test_metrics.json
$OUTPUT_DIR/fixture/summary.json
```

Значения метрик smoke-запуска не интерпретируются: модель сделала только один
шаг. Проверяется работоспособность полного вычислительного пути.

При загрузке исходного checkpoint допустим отчёт об `UNEXPECTED` ключах
`head.*` или `decoder.*`: это старая MLM-голова. Multi-head trainer загружает
общий ModernBERT encoder и создаёт новые classification heads.

## 5. Полное обучение

```bash
export CORPUS_DIR=/home/<user>/airi_artifacts/corpus
export BASE_MODEL=/home/<user>/airi_artifacts/modernbert
export DIRECTIONAL_JSONL=/home/<user>/airi_artifacts/directional/directional_disagreements_v2.jsonl
export OUTPUT_DIR=/home/<user>/airi_runs/multitask_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUTPUT_DIR")"

DEVICE=cuda \
PRECISION=bf16 \
INPUT_MODE=marked_pair \
MAX_LENGTH=512 \
BATCH_SIZE=32 \
RANKING_BATCH_SIZE=8 \
GRAD_ACCUM=1 \
EPOCHS=3 \
LR=2e-5 \
NUM_WORKERS=8 \
bash mvp/run_multitask_cross_encoder_h100.sh 2>&1 | tee "$OUTPUT_DIR.console.log"
```

Если возникает CUDA OOM, сначала уменьшите `BATCH_SIZE` до 16 и задайте
`GRAD_ACCUM=2`. Эффективный batch останется равным 32.

## 6. Что именно оптимизируется

```text
L = L_preserve
  + 0.30 * L_nli(A -> B)
  + 0.30 * L_nli(B -> A)
  + 0.15 * L_relation
  + 0.25 * L_ranking
  + 0.05 * L_consistency
```

- `L_preserve` использует все бинарные пары.
- Directional и relation losses маскируются для строк без соответствующих меток.
- Weak/rule-based строки получают вес `0.4`.
- Ranking использует только `ranking_train.jsonl`, построенный из train.
- Checkpoint пока выбирается преимущественно по preservation validation score,
  потому что directional validation содержит только три строки.

## 7. Проверка результата

После обучения должны существовать:

```text
$OUTPUT_DIR/best/
$OUTPUT_DIR/training_metadata.json
$OUTPUT_DIR/metrics.jsonl
$OUTPUT_DIR/test_metrics.json
```

Проверить новый JSONL с парами:

```bash
python mvp/score_multitask_pairs.py \
  --checkpoint "$OUTPUT_DIR/best" \
  --input /path/to/pairs.jsonl \
  --out /path/to/pairs.scored.jsonl \
  --summary-out /path/to/pairs.scored.summary.json \
  --device cuda
```

Основные поля результата: `multitask_preserved_probability`,
`multitask_label`, `multitask_a_to_b`, `multitask_b_to_a`,
`multitask_relation` и `multitask_relation_agreement`.

## 8. Критерии содержательной оценки

Полный запуск нельзя оценивать только по общей accuracy. Зафиксируйте:

- macro-F1, balanced accuracy, PR-AUC и calibration preservation-head;
- ranking accuracy на hard negatives;
- метрики отдельно по `entity_type`, `pair_kind`, domain и coarse group;
- ошибки alias positives и близких hard negatives;
- NLI-head как диагностику, пока directional-корпус не расширен.
