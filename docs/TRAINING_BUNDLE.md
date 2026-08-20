# AIRI MARS H100 training bundle v1

Архив `airi_mars_training_bundle_v1_20260820.zip` содержит все внешние
артефакты, необходимые для запуска уже подготовленного multi-head trainer:
канонический корпус, ranking-пары, directional-разметку и локальный ModernBERT.
API-ключи и сетевой доступ не нужны.

## Содержимое архива

```text
airi_mars_training_bundle_v1/
├── README.md
├── SHA256SUMS
├── corpus/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── ranking_train.jsonl
│   ├── manifest.json
│   └── ranking_summary.json
├── directional/
│   └── directional_disagreements_v2.jsonl
└── modernbert/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

В архив не включены старые `optimizer.pt`, `scheduler.pt`, `rng_state.pth`,
`trainer_state.json` и `training_args.bin`: для нового обучения они не нужны.

## Распаковка на сервере

Работаем только внутри `/home/fvaluev`:

```bash
cd /home/fvaluev
unzip airi_mars_training_bundle_v1_20260820.zip
cd /home/fvaluev/airi_mars_training_bundle_v1
shasum -a 256 -c SHA256SUMS
```

Все строки проверки должны завершиться `OK`.

Код располагается отдельно:

```text
/home/fvaluev/AIRI-MARS
```

Итоговая структура:

```text
/home/fvaluev/
├── AIRI-MARS/
└── airi_mars_training_bundle_v1/
```

## Пути для runner-ов

```bash
cd /home/fvaluev/AIRI-MARS

export AIRI_ARTIFACTS=/home/fvaluev/airi_mars_training_bundle_v1
export CORPUS_DIR="$AIRI_ARTIFACTS/corpus"
export BASE_MODEL="$AIRI_ARTIFACTS/modernbert"
export DIRECTIONAL_JSONL="$AIRI_ARTIFACTS/directional/directional_disagreements_v2.jsonl"
```

Именно эти три переменные подхватывают `run_multitask_smoke.sh` и
`run_multitask_cross_encoder_h100.sh`.

## Проверка одного шага

```bash
export OUTPUT_DIR=/home/fvaluev/airi_runs/smoke_$(date +%Y%m%d_%H%M%S)

DEVICE=cuda \
PRECISION=bf16 \
MAX_LENGTH=128 \
bash mvp/run_multitask_smoke.sh
```

Ожидаемый финал:

```text
SMOKE TEST PASSED: exactly one train batch completed
```

Smoke-runner использует восемь настоящих строк каждого split, оба бинарных
класса, exact directional-пример и одну ranking-пару. Он проверяет forward,
backward, optimizer step, validation, сохранение, повторную загрузку и test.
Значения метрик после одного шага не интерпретируются.

## Полное обучение

```bash
export OUTPUT_DIR=/home/fvaluev/airi_runs/multitask_$(date +%Y%m%d_%H%M%S)
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

При CUDA OOM используйте `BATCH_SIZE=16 GRAD_ACCUM=2`.

## Что будет обучено

Будет дообучен один общий ModernBERT encoder и четыре головы:

1. `preservation`: `preserved / changed` для основной оценки допустимости.
2. `A -> B`: `entailment / neutral / contradiction`.
3. `B -> A`: те же классы в обратном направлении.
4. `relation`: equivalence, generalization, specialization, substitution или
   contradiction.

Ranking не создаёт отдельную голову. Он использует preservation logit и учит
ставить правильную подстановку выше близкого сложного отрицательного кандидата.

В текущем корпусе directional-разметки мало: 30 train, 3 validation и 5 test
строк. Поэтому первый содержательно полезный результат — calibrated
preservation + ranking scorer. Directional и relation heads обучаются и
сохраняются, но пока рассматриваются как экспериментальная диагностика, а не
как самостоятельно доказанная production-модель.

Результат обучения появится в:

```text
$OUTPUT_DIR/best/
$OUTPUT_DIR/training_metadata.json
$OUTPUT_DIR/metrics.jsonl
$OUTPUT_DIR/test_metrics.json
```
