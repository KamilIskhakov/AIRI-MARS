#!/usr/bin/env python3
"""Inspect local model.pt checkpoints and show why they are custom."""

from __future__ import annotations

import json
from pathlib import Path

import torch


DEFAULTS = [
    "models/h1/cross_attention/cnn_dailymail+xsum+multi_news+samsum/cross_attention/checkpoint-120000/model.pt",
    "models/h1/dual_encoder/cnn_dailymail+xsum+multi_news+samsum/dual_encoder/checkpoint-120000/model.pt",
    "models/h1/entity_list/cnn_dailymail+xsum+multi_news+samsum/entity_list/final/model.pt",
]


def main() -> None:
    for raw in DEFAULTS:
        path = Path(raw)
        if not path.exists():
            continue
        print(f"\n## {path}")
        state = torch.load(path, map_location="cpu")
        print(f"type={type(state)} keys={len(state) if isinstance(state, dict) else 'n/a'}")
        if not isinstance(state, dict):
            continue
        prefixes: dict[str, int] = {}
        for key in state.keys():
            prefix = key.split(".", 1)[0]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        print("prefixes=" + json.dumps(prefixes, ensure_ascii=False, indent=2))
        print("first_keys:")
        for key in list(state.keys())[:20]:
            value = state[key]
            shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
            print(f"  {key}: {shape}")
        print("last_keys:")
        for key in list(state.keys())[-12:]:
            value = state[key]
            shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
            print(f"  {key}: {shape}")


if __name__ == "__main__":
    main()
