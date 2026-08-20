#!/usr/bin/env python3
"""Export entity inventory rows from SQLite to JSONL for LLM tagging."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def parse_json(raw: str):
    try:
        return json.loads(raw or "{}")
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--group", action="append")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--per-group", type=int, default=100)
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = []
    if args.balanced:
        groups = args.group or ["proper_name", "numeric", "domain_term", "common_entity"]
        for group in groups:
            rows.extend(
                conn.execute(
                    """
                    SELECT id, surface, heuristic_group, heuristic_type, context_policy,
                           mention_count, observed_types_json, domains_json, datasets_json
                    FROM entities
                    WHERE heuristic_group = ?
                    ORDER BY mention_count DESC
                    LIMIT ?
                    """,
                    (group, args.per_group),
                ).fetchall()
            )
    else:
        where = ""
        params: list[object] = []
        if args.group:
            placeholders = ",".join("?" for _ in args.group)
            where = f"WHERE heuristic_group IN ({placeholders})"
            params.extend(args.group)
        params.append(args.limit)
        rows = conn.execute(
            f"""
            SELECT id, surface, heuristic_group, heuristic_type, context_policy,
                   mention_count, observed_types_json, domains_json, datasets_json
            FROM entities
            {where}
            ORDER BY mention_count DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows[: args.limit] if args.balanced else rows:
            item = {
                "entity_id": str(row["id"]),
                "entity": row["surface"],
                "count": row["mention_count"],
                "observed_types": parse_json(row["observed_types_json"]),
                "domains": parse_json(row["domains_json"]),
                "datasets": parse_json(row["datasets_json"]),
                "heuristic_group": row["heuristic_group"],
                "heuristic_context_policy": row["context_policy"],
                "heuristic_type": row["heuristic_type"],
                "examples": [],
            }
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"exported={len(rows[: args.limit] if args.balanced else rows)} wrote={args.out}")
    conn.close()


if __name__ == "__main__":
    main()
