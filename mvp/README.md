# MVP: Contextual Entity Substitution Scorer

Этот MVP не делает новый суммаризатор. Он готовит данные и обучает отдельный scorer:

```text
(C[e], C[e_hat]) -> допустима ли подстановка e_hat вместо e в данном контексте
```

Веса из репозитория используются так:

- `models/h1/merged/.../checkpoint-40000` — локальная ModernBERT-база для нового cross-encoder scorer-а.
- `models/h1/cross_attention`, `dual_encoder`, `entity_list` — кастомные `model.pt`; для них есть `inspect_checkpoints.py`, но без исходных классов они не грузятся напрямую как `AutoModel`.

## 1. Окружение

```bash
python3 -m venv .venv-mvp
source .venv-mvp/bin/activate
pip install -r requirements-mvp.txt
```

## 2. Smoke dataset

Лучше начать с 200 строк:

```bash
python mvp/prepare_pairs.py \
  --dataset-dir data/mars_test_200_split/test \
  --out data/mvp_pairs_mars200.jsonl \
  --max-rows 200 \
  --negatives-per-positive 3
```

Для полного typed BillSum:

```bash
python mvp/prepare_pairs.py \
  --dataset-dir data/billsum_with_mask/train \
  --out data/mvp_pairs_billsum_train.jsonl \
  --max-rows 5000 \
  --negatives-per-positive 3
```

## 3. Train cross-encoder scorer

Для текущей LLM-разметки proper names после завершения пайплайна будет файл:

```text
data/proper_name_agent_annotation_strict_12k/train_consensus.jsonl
```

На машине с RTX 4090 запускай готовый runner:

```bash
bash mvp/run_cross_encoder_4090.sh
```

Полезные параметры:

```bash
BASE_MODEL=microsoft/deberta-v3-base \
EPOCHS=3 \
BATCH_SIZE=24 \
MAX_LENGTH=512 \
PRECISION=bf16 \
bash mvp/run_cross_encoder_4090.sh
```

По умолчанию runner сначала использует локальный чекпойнт
`models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000`,
если он есть, иначе берет `microsoft/deberta-v3-base`.

Основной режим `marked_pair`: модель получает два текста, где исходная сущность и кандидат помечены токенами `[E1]...[/E1]` и `[E2]...[/E2]`.
Это сейчас главный вариант для MVP, потому что scorer видит полный контекст до и после замены.

Для сравнения можно прогнать альтернативный режим:

```bash
RUN_ENTITY_QUERY=1 bash mvp/run_cross_encoder_4090.sh
```

Результаты лежат в `runs/cross_encoder/...`:

- `best/` — лучший чекпойнт по `macro_f1` на validation;
- `final/` — последний чекпойнт;
- `training_metadata.json` — размеры split-ов, баланс классов, типы пар;
- `metrics.jsonl` — метрики по эпохам;
- `test_metrics.json` — финальная проверка на held-out entity split.

Старый smoke-пример на маленьких парах:

```bash
python mvp/train_cross_encoder.py \
  --train-jsonl data/mvp_pairs_mars200.jsonl \
  --base-model models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000 \
  --output-dir models/mvp/contextual_substitution_scorer \
  --epochs 1 \
  --batch-size 2 \
  --max-length 512
```

На Mac без GPU/с ограниченной памятью начни с `--batch-size 1 --max-length 384`.

## 4. Улучшенная аугментация пар

`prepare_pairs.py` делает самый простой набор. Для более честного MVP лучше использовать `augment_pairs.py`:

```bash
python mvp/augment_pairs.py \
  --dataset-dir data/mars_test_200_split/test \
  --out data/mvp_augmented_mars200.jsonl \
  --agent-review-out data/mvp_agent_review_mars200.jsonl \
  --max-rows 200 \
  --negatives-per-entity 3 \
  --review-candidates-per-entity 1
```

Он генерирует:

- `positive_identity`: исходная сущность вместо самой себя;
- `positive_surface`: простые surface-варианты;
- `positive_alias`: строгие алиасы из `mvp/resources/aliases.json`;
- `negative_same_type`: другая сущность того же типа;
- `negative_numeric_or_date_perturbation`: близкая, но фактологически другая дата/сумма/число;
- `agent-review` задачи для контекстно-спорных кандидатов, например `America` для `United States`.
- `agent_hard_same_type_candidate`: кандидаты того же типа без автолейбла; их надо разметить агентом, чтобы получить более честные хорошие/плохие пары.

Файл `agent-review` не надо сразу скармливать в train. Его надо разметить агентом/LLM/человеком в `label = 1 / 0.5 / 0`.

## 5. Инвентарь сущностей и Mistral-разметка

Перед генерацией хороших/плохих подстановок можно собрать массив всех уникальных masked-сущностей:

```bash
python mvp/extract_entity_inventory.py \
  --dataset-dir data/mars_test_200_split/test \
  --out data/entity_inventory_mars200.jsonl \
  --summary-out data/entity_inventory_mars200.summary.json \
  --max-rows-per-dataset 200
```

Для нескольких доменов:

```bash
python mvp/extract_entity_inventory.py \
  --dataset-dir data/untyped/cnn_dailymail_with_mask/train \
  --dataset-dir data/untyped/xsum_with_mask/train \
  --dataset-dir data/untyped/samsum_with_mask/train \
  --dataset-dir data/untyped/billsum_with_mask/train \
  --out data/entity_inventory_sample.jsonl \
  --summary-out data/entity_inventory_sample.summary.json \
  --max-rows-per-dataset 1000
```

Разметка типов через Mistral с Pydantic-схемой:

```bash
export MISTRAL_API_KEY=...
python mvp/tag_entities_mistral.py \
  --inventory data/entity_inventory_mars200.jsonl \
  --out data/entity_tags_mars200.jsonl \
  --model ministral-8b-latest \
  --batch-size 30 \
  --resume
```

Сначала можно посмотреть промпт и JSON Schema без вызова API:

```bash
python mvp/tag_entities_mistral.py \
  --inventory data/entity_inventory_mars200.jsonl \
  --out /tmp/mistral_tagging_dry_run.json \
  --limit 10 \
  --dry-run
```

OpenAI-compatible провайдеры запускаются через серверный `response_format=json_schema`.
Схема генерируется из `EntityTagBatch`/`EntityTag`, а локальная Pydantic-проверка остается как страховка соответствия входным `entity_id` и `entity`:

```bash
python mvp/tag_entities_providers.py \
  --provider groq \
  --inventory data/entity_inventory_probe9_stratified.jsonl \
  --out data/entity_tags_probe9.groq_gptoss.jsonl \
  --limit 9 \
  --batch-size 3 \
  --schema-mode json_schema
```

Мини-проверка на `data/entity_inventory_probe9_stratified.jsonl`:

- `mistral-small-latest`: прошел Pydantic `chat.parse`, качество адекватное.
- `groq` + `openai/gpt-oss-120b`: прошел `json_schema`, качество адекватное.
- `cerebras` + `gpt-oss-120b`: прошел `json_schema`; для совместимости API-схема очищается от `maxLength/minimum/maximum`, но финальная локальная Pydantic-проверка остается полной.
- `openrouter` + `openai/gpt-4.1-mini`: прошел `json_schema`, но на probe были странные `confidence=0`, поэтому пока не основной теггер.
- `groq_fast`/`llama-3.1-8b-instant` и `groq`/`llama-3.3-70b-versatile`: не поддерживают `response_format=json_schema`, только менее строгий JSON mode.

Разметчик возвращает:

- `coarse_group`: `proper_name`, `numeric`, `common_entity`, `domain_term`, `ambiguous`, `junk`;
- `fine_type`: `PERSON`, `ORG`, `GPE`, `DATE`, `MONEY`, `COMMON_NOUN`, etc.;
- `context_policy`: `short_window`, `full_context`, `no_context_embedding`, `agent_review`, `drop`.

Идея политики:

- числа/даты/суммы: короткое окно 1-2 слова;
- конкретные имена, организации, места: полный локальный контекст;
- общие слова и термины: сначала embedding-кандидаты, потом проверка качества;
- спорное: агентная разметка с контекстом.

Если провайдер в structured output случайно меняет текст `entity`, скрипт восстанавливает исходное значение из inventory по `entity_id` и пишет запись в `*.warnings.jsonl`. Если батч совсем не выравнивается по `entity_id`, он ретраится; после исчерпания ретраев включается эвристический fallback, если не указан `--no-fallback-on-failure`.

## Common nouns: embedding retrieval

Полная актуальная схема валидации, генерации common/numeric данных, H100-абляций, ранжирования и двунаправленного NLI описана в `mvp/SEMANTIC_PIPELINE.md`.

Полностью офлайн multi-head запуск на H100:

```bash
PYTHON_BIN=/path/to/python DEVICE=cuda PRECISION=bf16 \
bash mvp/run_multitask_cross_encoder_h100.sh
```

Один encoder обучает preservation, два направления NLI и relation head. NLI-
потери применяются только к строкам с реальной направленной меткой; ranking loss
строится по positive/hard-negative кандидатам одного упоминания.

Для нарицательных сущностей используется отдельная ветка:

1. Лексическая очистка пула и проверка английских существительных через WordNet.
2. Повторная агентная типизация с восстановленным контекстом из SQLite.
3. Поиск кандидатов локальным `multilingual-e5-small` с высокими порогами: `>=0.97` и `0.90-0.97`.
4. Удаление identity и простых singular/plural пар до разметки.
5. Независимое контекстное решение Mistral и OpenRouter; в train попадает только консенсус.

WordNet устанавливается один раз:

```bash
python -m nltk.downloader wordnet
```

Поиск среди уже встречавшихся сущностей выполняет `generate_common_embedding_pairs.py`.
Чтобы расширить бедный положительный класс, `generate_common_wordnet_embedding_pairs.py` ищет соседей среди английских WordNet nouns. Косинус используется только для retrieval и не является меткой: `preserved/changed` назначают контекстные судьи.

Актуальный проверенный результат лежит в:

- `data/common_noun_embedding_final_20260714/final_audited.jsonl` — пары, голоса судей и итоговая метка;
- `data/common_noun_embedding_final_20260714/train_audited.jsonl` — train-ready строки;
- `data/common_noun_embedding_final_20260714/summary.json` — сводная статистика.

На текущем частотном пуле получено 116 пар: 106 `changed`, 3 `preserved`, 7 `uncertain`; идентичных исходной сущности кандидатов нет. Это качественный набор сложных отрицательных примеров, но положительный common-класс пока нужно расширять.

## 5.1. Единый запуск quality-first пайплайна

Для генерации common/numeric-кандидатов, расчёта `MLM-fit`, параллельного суда
Mistral + OpenRouter, консенсуса и сборки disjoint-корпуса используется:

```bash
PYTHON_BIN=python \
SECOND_JUDGE_PROVIDER=openrouter \
bash mvp/run_semantic_quality_pipeline.sh
```

Каждый запуск получает новую папку `data/semantic_pipeline_<timestamp>`; старые
файлы не удаляются. Сырые кандидаты не попадают в обучение: в corpus проходят
только бинарные строки с согласованными ответами двух судей. `uncertain` остаётся
в review. Перед дорогим запуском можно отдельно выполнить:

```bash
python mvp/audit_semantic_dataset.py \
  --input corpus=data/semantic_corpus_v1_20260819/all.jsonl \
  --out data/semantic_quality_audit.json \
  --sample-out data/semantic_quality_audit.samples.jsonl
```

Cross-encoder восстанавливает целевое упоминание в `model_left/model_right` по
`mask_idx` и проверяет его по candidate-контексту. Это важно для документов, где
одна и та же сущность встречается несколько раз: простого поиска строки было бы
недостаточно.

### Common без массовых LLM-вызовов

```bash
PYTHON_BIN=python \
VERY_HIGH_MIN=0.97 HARD_MIN=0.90 HARD_MAX=0.97 \
bash mvp/run_common_weak_generation.sh
```

Эта ветка полностью локальная:

- существующие `common_entity` берутся из SQLite-инвентаря;
- ModernBERT строит соседей по cosine similarity;
- морфологические варианты (`mosque -> mosques`) удаляются;
- прямые WordNet-синонимы получают слабую положительную метку;
- близкие кандидаты без общего synset в диапазоне `0.90-0.97` получают слабую
  метку hard negative;
- диапазон `0.82-0.90` подключается только явно через `INCLUDE_MEDIUM=1`.

Слабые метки не считаются золотым стандартом. Для контроля достаточно отправить
Mistral/OpenRouter небольшой стратифицированный сэмпл по `candidate_kind` и
проверить precision каждого правила; массово прогонять через LLM весь common-пул
не требуется.

## 6. Score/evaluate pairs

```bash
python mvp/score_pairs.py \
  --model-dir models/mvp/contextual_substitution_scorer \
  --input-jsonl data/mvp_pairs_mars200.jsonl \
  --output-jsonl data/mvp_pairs_mars200.scored.jsonl \
  --max-length 512
```

Для общей multi-head модели используется отдельный офлайн-скорер:

```bash
python mvp/score_multitask_pairs.py \
  --checkpoint models/semantic_multitask_<run>/best \
  --input data/pairs_to_score.jsonl \
  --out data/pairs_to_score.multitask.jsonl \
  --summary-out data/pairs_to_score.multitask.summary.json \
  --device cuda
```

Он возвращает вероятность сохранения смысла, решение `preserved/changed/uncertain`,
два направленных NLI-ответа и тип отношения. До расширения directional-разметки
решение preservation-head можно использовать для основной оценки и reject-band,
а NLI/relation-head следует считать диагностикой: в текущем фиксированном корпусе
точные направленные метки покрывают только 30 train, 3 validation и 5 test строк.

## 7. Inspect старых чекпойнтов

```bash
python mvp/inspect_checkpoints.py
```

## Что именно делает MVP

`prepare_pairs.py` строит пары:

- positive: `C[e]` против `C[e]` или простой surface-вариант той же сущности;
- negative: `C[e]` против `C[e_negative]`, где negative берется из других сущностей, по возможности того же `entity_type`.

Это базовый старт. Следующий шаг после MVP — добавить настоящие alias-позитивы и embedding retrieval top-K, чтобы cross-encoder учился не только identity matching, а именно контекстной эквивалентности.

`augment_pairs.py` — следующий слой. Он уже начинает строить хорошие/плохие примеры подстановок, связанные с исходной сущностью и типом сущности, а спорные контекстные случаи выносит в отдельную очередь для разметки.
