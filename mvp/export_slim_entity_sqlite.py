#!/usr/bin/env python3
"""Export a compact entity_inventory.sqlite subset for remote annotation."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


TABLES = ("datasets", "texts", "entities", "mentions")


def load_entity_ids(tag_files: list[Path], coarse_group: str, limit: int) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for path in tag_files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if coarse_group and row.get("coarse_group") != coarse_group:
                    continue
                entity_id = int(row["entity_id"])
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                ids.append(entity_id)
                if limit and len(ids) >= limit:
                    return ids
    return ids


def row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def insert_row(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    columns = list(row)
    placeholders = ",".join("?" for _ in columns)
    names = ",".join(columns)
    conn.execute(
        f"insert or ignore into {table} ({names}) values ({placeholders})",
        [row[column] for column in columns],
    )


def copy_schema(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    for table in TABLES:
        sql = src.execute(
            "select sql from sqlite_master where type = 'table' and name = ?",
            (table,),
        ).fetchone()[0]
        dst.execute(sql)
    for row in src.execute(
        "select sql from sqlite_master where type = 'index' and sql is not null and tbl_name in "
        "('datasets','texts','entities','mentions')"
    ):
        dst.execute(row[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, default=Path("data/entity_inventory.sqlite"))
    parser.add_argument("--out-db", required=True, type=Path)
    parser.add_argument("--tag-files", type=Path, nargs="+", required=True)
    parser.add_argument("--coarse-group", default="proper_name")
    parser.add_argument("--limit-entities", type=int, default=0)
    parser.add_argument("--mentions-per-entity", type=int, default=12)
    parser.add_argument("--commit-every", type=int, default=1000)
    args = parser.parse_args()

    entity_ids = load_entity_ids(args.tag_files, args.coarse_group, args.limit_entities)
    if args.out_db.exists():
        raise FileExistsError(args.out_db)
    args.out_db.parent.mkdir(parents=True, exist_ok=True)

    src = sqlite3.connect(f"file:{args.source_db}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    dst = sqlite3.connect(args.out_db)
    dst.row_factory = sqlite3.Row
    dst.execute("pragma journal_mode=off")
    dst.execute("pragma synchronous=off")
    copy_schema(src, dst)

    for row in src.execute("select * from datasets"):
        insert_row(dst, "datasets", row_dict(row))

    copied_mentions = 0
    copied_texts: set[int] = set()
    copied_entities = 0
    for idx, entity_id in enumerate(entity_ids, start=1):
        entity = src.execute("select * from entities where id = ?", (entity_id,)).fetchone()
        if not entity:
            continue
        insert_row(dst, "entities", row_dict(entity))
        copied_entities += 1
        mentions = src.execute(
            "select * from mentions where entity_id = ? limit ?",
            (entity_id, args.mentions_per_entity),
        ).fetchall()
        for mention in mentions:
            mention_row = row_dict(mention)
            text_id = int(mention_row["text_id"])
            if text_id not in copied_texts:
                text = src.execute("select * from texts where id = ?", (text_id,)).fetchone()
                if text:
                    insert_row(dst, "texts", row_dict(text))
                    copied_texts.add(text_id)
            insert_row(dst, "mentions", mention_row)
            copied_mentions += 1
        if idx % args.commit_every == 0:
            dst.commit()
            print(
                json.dumps(
                    {
                        "processed_entities": idx,
                        "copied_entities": copied_entities,
                        "copied_texts": len(copied_texts),
                        "copied_mentions": copied_mentions,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    dst.commit()
    dst.execute("pragma optimize")
    summary = {
        "requested_entities": len(entity_ids),
        "copied_entities": copied_entities,
        "copied_texts": len(copied_texts),
        "copied_mentions": copied_mentions,
        "out_db": str(args.out_db),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
