# AIRI MARS: semantic entity substitution

Код для обучения ModernBERT cross-encoder, который оценивает, сохраняет ли
подстановка сущности смысл исходного утверждения.

Модель получает два полноценных текста:

```text
A = контекст с исходной сущностью
B = тот же контекст с кандидатом
```

Основная голова решает `preserved / changed`. Дополнительные головы оценивают
`A -> B`, `B -> A` и тип отношения; ranking loss учит ставить корректного
кандидата выше близкого, но неверного.

## Быстрый старт на H100

Точная инструкция, структура внешних артефактов, one-step smoke test и полный
запуск описаны в [docs/H100_TRAINING.md](docs/H100_TRAINING.md).
Для готового ZIP с корпусом и весами используйте
[docs/TRAINING_BUNDLE.md](docs/TRAINING_BUNDLE.md).

После размещения корпуса и локальных весов первым делом выполните:

```bash
CORPUS_DIR=/absolute/path/to/semantic_corpus_v1_20260819 \
BASE_MODEL=/absolute/path/to/modernbert_checkpoint \
DIRECTIONAL_JSONL=/absolute/path/to/directional_disagreements_v2.jsonl \
DEVICE=cuda PRECISION=bf16 \
bash mvp/run_multitask_smoke.sh
```

Smoke-команда строит временный срез из реального корпуса и выполняет ровно один
optimizer step, включая preservation, directional/relation и ranking branches.
Только после её успеха запускайте полное обучение.

## Что находится в Git

- `mvp/train_multitask_cross_encoder.py` — multi-head обучение;
- `mvp/run_multitask_cross_encoder_h100.sh` — полный H100 runner;
- `mvp/run_multitask_smoke.sh` — обязательная проверка одного шага;
- `mvp/score_multitask_pairs.py` — offline inference;
- `mvp/run_semantic_offline_h100.sh` — генерация common/numeric и обучение без API;
- `presentations/` — актуальная Typst-презентация решения.

Датасеты, модельные веса, API-ключи и результаты запусков намеренно исключены
из Git. Канонический корпус занимает около 300 МБ, а используемый checkpoint
ModernBERT — около 575 МБ без optimizer state.

## Текущий статус данных

- 24 935 бинарных пар;
- 19 947 train / 2 494 validation / 2 494 test;
- 10 975 ranking-пар, построенных только из train;
- 30 / 3 / 5 directional-строк в train / validation / test;
- 24 826 proper-name и 109 common-пар в каноническом корпусе;
- numeric-генератор готов, но numeric ещё не входит в канонический корпус.

Directional-данных пока недостаточно для выбора модели по NLI-F1. Первый
полезный полный запуск оценивается прежде всего по preservation и ranking.
