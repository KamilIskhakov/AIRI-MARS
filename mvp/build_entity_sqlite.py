#!/usr/bin/env python3
"""Build a SQLite inventory of masked entities, source texts, and mentions."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from augment_pairs import infer_entity_type, norm_key
from extract_entity_inventory import (
    default_context_policy,
    domain_from_path,
    heuristic_group,
    local_word_window,
)
from prepare_pairs import MASK_RE, crop_around_entity, iter_arrow_rows, normalize_space, replace_nth_mask


def dataset_name_from_dir(dataset_dir: Path) -> tuple[str, str]:
    split = dataset_dir.name
    root = dataset_dir.parent
    if split not in {"train", "validation", "test", "ca_test"}:
        split = "default"
        root = dataset_dir
    return root.name, split


def init_db(conn: sqlite3.Connection, reset: bool) -> None:
    if reset:
        conn.executescript(
            """
            DROP TABLE IF EXISTS mentions;
            DROP TABLE IF EXISTS entities;
            DROP TABLE IF EXISTS texts;
            DROP TABLE IF EXISTS datasets;
            DROP TABLE IF EXISTS run_metadata;
            """
        )

    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;

        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            split TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            domain TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS texts (
            id INTEGER PRIMARY KEY,
            dataset_id INTEGER NOT NULL,
            row_idx INTEGER NOT NULL,
            source_id TEXT,
            title TEXT,
            original_text TEXT,
            summary TEXT,
            masked_text TEXT NOT NULL,
            mask_count INTEGER NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES datasets(id),
            UNIQUE(dataset_id, row_idx)
        );

        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            surface_key TEXT NOT NULL UNIQUE,
            surface TEXT NOT NULL,
            heuristic_group TEXT NOT NULL,
            heuristic_type TEXT NOT NULL,
            context_policy TEXT NOT NULL,
            mention_count INTEGER NOT NULL DEFAULT 0,
            observed_types_json TEXT NOT NULL DEFAULT '{}',
            domains_json TEXT NOT NULL DEFAULT '{}',
            datasets_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS mentions (
            id INTEGER PRIMARY KEY,
            entity_id INTEGER NOT NULL,
            text_id INTEGER NOT NULL,
            mask_idx INTEGER NOT NULL,
            observed_type TEXT NOT NULL,
            short_context TEXT NOT NULL,
            full_context TEXT NOT NULL,
            FOREIGN KEY(entity_id) REFERENCES entities(id),
            FOREIGN KEY(text_id) REFERENCES texts(id),
            UNIQUE(text_id, mask_idx)
        );

        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_entities_surface ON entities(surface);
        CREATE INDEX IF NOT EXISTS idx_entities_group ON entities(heuristic_group);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(heuristic_type);
        CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_id);
        CREATE INDEX IF NOT EXISTS idx_mentions_text ON mentions(text_id);
        CREATE INDEX IF NOT EXISTS idx_texts_dataset ON texts(dataset_id);
        """
    )


def upsert_dataset(conn: sqlite3.Connection, dataset_dir: Path) -> int:
    name, split = dataset_name_from_dir(dataset_dir)
    domain = domain_from_path(dataset_dir)
    path = str(dataset_dir)
    conn.execute(
        """
        INSERT INTO datasets(name, split, path, domain)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            name=excluded.name,
            split=excluded.split,
            domain=excluded.domain
        """,
        (name, split, path, domain),
    )
    row = conn.execute("SELECT id FROM datasets WHERE path = ?", (path,)).fetchone()
    return int(row[0])


def insert_text(
    conn: sqlite3.Connection,
    dataset_id: int,
    row_idx: int,
    row: dict[str, Any],
    mask_count: int,
    store_texts: bool,
) -> int:
    original_text = row.get("original_text") if store_texts else None
    summary = row.get("summary") if store_texts else None
    masked_text = row.get("masked_text") or ""
    title = row.get("title") if store_texts else None
    source_id = str(row.get("id")) if row.get("id") is not None else None

    conn.execute(
        """
        INSERT INTO texts(dataset_id, row_idx, source_id, title, original_text, summary, masked_text, mask_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id, row_idx) DO UPDATE SET
            source_id=excluded.source_id,
            title=excluded.title,
            original_text=excluded.original_text,
            summary=excluded.summary,
            masked_text=excluded.masked_text,
            mask_count=excluded.mask_count
        """,
        (dataset_id, row_idx, source_id, title, original_text, summary, masked_text, mask_count),
    )
    text_id = conn.execute(
        "SELECT id FROM texts WHERE dataset_id = ? AND row_idx = ?",
        (dataset_id, row_idx),
    ).fetchone()[0]
    return int(text_id)


def upsert_entity(conn: sqlite3.Connection, surface: str, observed_type: str) -> int:
    key = norm_key(surface)
    group = heuristic_group(surface, observed_type)
    policy = default_context_policy(group)
    conn.execute(
        """
        INSERT INTO entities(surface_key, surface, heuristic_group, heuristic_type, context_policy)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(surface_key) DO NOTHING
        """,
        (key, surface, group, observed_type, policy),
    )
    entity_id = conn.execute("SELECT id FROM entities WHERE surface_key = ?", (key,)).fetchone()[0]
    return int(entity_id)


def process_dataset(
    conn: sqlite3.Connection,
    dataset_dir: Path,
    max_rows: int | None,
    store_texts: bool,
    store_contexts: bool,
    short_context_words: int,
    full_context_chars: int,
    commit_every: int,
) -> dict[str, int]:
    dataset_id = upsert_dataset(conn, dataset_dir)
    stats = Counter()
    observed_types: dict[int, Counter[str]] = {}
    domains: dict[int, Counter[str]] = {}
    datasets: dict[int, Counter[str]] = {}
    entity_cache: dict[str, int] = {}
    dataset_name, split = dataset_name_from_dir(dataset_dir)
    domain = domain_from_path(dataset_dir)

    for row_idx, row in enumerate(iter_arrow_rows(dataset_dir, max_rows)):
        masked_text = row.get("masked_text") or ""
        words = row.get("demasked_words") or []
        types = row.get("entity_types") or []
        mask_count = len(MASK_RE.findall(masked_text))
        usable = min(mask_count, len(words))

        text_id = insert_text(conn, dataset_id, row_idx, row, mask_count, store_texts)
        stats["rows"] += 1
        stats["masks"] += usable

        for mask_idx in range(usable):
            surface = normalize_space(str(words[mask_idx]))
            if not surface:
                continue
            given = str(types[mask_idx]) if mask_idx < len(types) and types[mask_idx] else None
            observed_type = infer_entity_type(surface, given)
            surface_key = norm_key(surface)
            entity_id = entity_cache.get(surface_key)
            if entity_id is None:
                entity_id = upsert_entity(conn, surface, observed_type)
                entity_cache[surface_key] = entity_id
            if store_contexts:
                filled = replace_nth_mask(masked_text, mask_idx, surface)
                short_context = local_word_window(filled, surface, short_context_words)
                full_context = crop_around_entity(filled, surface, full_context_chars)
            else:
                short_context = ""
                full_context = ""
            conn.execute(
                """
                INSERT INTO mentions(entity_id, text_id, mask_idx, observed_type, short_context, full_context)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(text_id, mask_idx) DO UPDATE SET
                    entity_id=excluded.entity_id,
                    observed_type=excluded.observed_type,
                    short_context=excluded.short_context,
                    full_context=excluded.full_context
                """,
                (
                    entity_id,
                    text_id,
                    mask_idx,
                    observed_type,
                    short_context,
                    full_context,
                ),
            )
            observed_types.setdefault(entity_id, Counter())[observed_type] += 1
            domains.setdefault(entity_id, Counter())[domain] += 1
            datasets.setdefault(entity_id, Counter())[f"{dataset_name}/{split}"] += 1
            stats["mentions"] += 1

        if stats["rows"] % commit_every == 0:
            conn.commit()
            print(f"{dataset_dir}: rows={stats['rows']} mentions={stats['mentions']}", flush=True)

    conn.commit()
    update_entity_counters(conn, observed_types, domains, datasets)
    conn.commit()
    return dict(stats)


def merge_counter_json(old_json: str, delta: Counter[str]) -> str:
    merged = Counter(json.loads(old_json or "{}"))
    merged.update(delta)
    return json.dumps(dict(merged.most_common()), ensure_ascii=False)


def update_entity_counters(
    conn: sqlite3.Connection,
    observed_types: dict[int, Counter[str]],
    domains: dict[int, Counter[str]],
    datasets: dict[int, Counter[str]],
) -> None:
    entity_ids = set(observed_types) | set(domains) | set(datasets)
    for entity_id in entity_ids:
        row = conn.execute(
            """
            SELECT mention_count, observed_types_json, domains_json, datasets_json
            FROM entities WHERE id = ?
            """,
            (entity_id,),
        ).fetchone()
        if not row:
            continue
        mention_count, observed_json, domains_json, datasets_json = row
        delta_mentions = sum(observed_types.get(entity_id, Counter()).values())
        conn.execute(
            """
            UPDATE entities
            SET mention_count = ?,
                observed_types_json = ?,
                domains_json = ?,
                datasets_json = ?
            WHERE id = ?
            """,
            (
                int(mention_count) + delta_mentions,
                merge_counter_json(observed_json, observed_types.get(entity_id, Counter())),
                merge_counter_json(domains_json, domains.get(entity_id, Counter())),
                merge_counter_json(datasets_json, datasets.get(entity_id, Counter())),
                entity_id,
            ),
        )


def write_metadata(conn: sqlite3.Connection, args: argparse.Namespace, stats: dict[str, Any]) -> None:
    payload = {
        "dataset_dirs": [str(x) for x in args.dataset_dir],
        "store_texts": args.store_texts,
        "store_contexts": args.store_contexts,
        "max_rows_per_dataset": args.max_rows_per_dataset,
        "stats": stats,
    }
    conn.execute(
        """
        INSERT INTO run_metadata(key, value)
        VALUES ('last_run', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (json.dumps(payload, ensure_ascii=False, indent=2),),
    )


def print_summary(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
    entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    mentions = conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
    print(f"sqlite_summary texts={rows} entities={entities} mentions={mentions}")
    print("by_group")
    for row in conn.execute(
        "SELECT heuristic_group, COUNT(*) FROM entities GROUP BY heuristic_group ORDER BY COUNT(*) DESC"
    ):
        print(f"  {row[0]}: {row[1]}")
    print("by_type")
    for row in conn.execute(
        "SELECT heuristic_type, COUNT(*) FROM entities GROUP BY heuristic_type ORDER BY COUNT(*) DESC LIMIT 20"
    ):
        print(f"  {row[0]}: {row[1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", action="append", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument("--store-texts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store-contexts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--short-context-words", type=int, default=2)
    parser.add_argument("--full-context-chars", type=int, default=700)
    parser.add_argument("--commit-every", type=int, default=1000)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    init_db(conn, args.reset)
    max_rows = args.max_rows_per_dataset or None
    all_stats: dict[str, Any] = {}

    try:
        for dataset_dir in args.dataset_dir:
            stats = process_dataset(
                conn,
                dataset_dir,
                max_rows,
                args.store_texts,
                args.store_contexts,
                args.short_context_words,
                args.full_context_chars,
                args.commit_every,
            )
            all_stats[str(dataset_dir)] = stats
        write_metadata(conn, args, all_stats)
        conn.commit()
        print_summary(conn)
        print(f"wrote {args.db}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
