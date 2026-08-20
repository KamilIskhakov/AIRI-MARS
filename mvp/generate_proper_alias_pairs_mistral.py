#!/usr/bin/env python3
"""Generate proper-name substitution candidates with a context-aware Mistral agent.

The agent creates both:
- same-entity aliases/surface variants (good candidates);
- plausible hard negatives of the same role/type/domain (bad candidates).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from audit_entity_tags_and_probe import connect_ro, sample_mentions
from prepare_pairs import replace_nth_mask
from run_small_substitution_probe import fill_contexts
from tag_entities_mistral import load_env_file


class GoodCandidate(BaseModel):
    candidate: str = Field(min_length=1, max_length=160)
    relation: Literal[
        "alias",
        "abbreviation",
        "acronym",
        "official_variant",
        "short_name",
        "translated_name",
        "spelling_variant",
        "person_short_name",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=240)


class BadCandidate(BaseModel):
    candidate: str = Field(min_length=1, max_length=160)
    relation: Literal[
        "same_role_different_entity",
        "same_domain_different_entity",
        "same_location_type",
        "same_org_type",
        "same_person_role",
        "near_alias_but_wrong",
        "context_plausible_wrong_entity",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    why_plausible: str = Field(max_length=240)
    why_wrong: str = Field(max_length=240)


class AliasItem(BaseModel):
    item_id: str = Field(min_length=1, max_length=80)
    entity_id: str
    entity: str
    good_candidates: list[GoodCandidate]
    bad_candidates: list[BadCandidate]


class AliasBatch(BaseModel):
    items: list[AliasItem]


SYSTEM_PROMPT = """You generate contextual substitution candidates for named entities.

Generate two lists:

1. good_candidates:
   Candidates that refer to the SAME real-world entity as the source entity in the given context.

2. bad_candidates:
   HARD negatives: close but wrong substitutions. They must be factually incorrect, but they must also be close enough that a weak model could confuse them with the source entity.

Good examples:
- "U.S. Department of Justice" -> "DOJ" or "Justice Department"
- "University of California, Berkeley" -> "UC Berkeley"
- "the Coppa Italia" -> "the Italian Cup"
- "United States" -> "U.S." if the context still clearly means the country
- "Robert Downey Jr." -> "Downey" only if the context makes the person unambiguous

Hard negative examples:
- In a newspaper/source context: "The Columbus Dispatch" -> "The New York Times", not "Red Lobster".
- In a university context: "Oregon State" -> "UCLA", not "16GB".
- In a government department context: "Department of Justice" -> "Department of Homeland Security".
- In a person/sports context: replace with another plausible person in the same sport/role, not a random unrelated person.
- In a city/location context: replace with another plausible city/location of the same type.
- In a law context: "Energy Policy Act" -> "Energy Policy and Conservation Act", not an unrelated criminal or agriculture law.
- In a committee context: "House Budget Committee" -> "Senate Finance Committee", not a federal agency.

Forbidden:
- Do not return the same literal string with only case/whitespace changes.
- Do not return easy absurd negatives.
- Do not return generic descriptors such as "the company", "the city", "the man".
- Do not invent aliases that are not standard or strongly supported by context.
- Do not broaden a named entity into a generic class or activity, e.g. "the Police Department" -> "local law enforcement".
- Do not replace an institution with an ambiguous place shorthand, e.g. "University of Edinburgh" -> "Edinburgh".
- Do not replace an event/competition with a related object or stage, e.g. "UEFA Super Cup" -> "UEFA Super Cup trophy" or "UEFA Super Cup final".
- Do not add or remove a disambiguating year unless the shorter or longer form is a standard canonical alias for the same entity.
- For good_candidates, do not return descriptive noun phrases that merely describe the entity,
  e.g. "the Air Traffic Controllers Union" for "the National Air Traffic Controllers Association".
- Legal citations and public-law numbers are allowed as good candidates only when they unambiguously identify the same law in the context.
- For bad_candidates, prefer candidates from the provided candidate_pool.
- For bad_candidates, do not use merely "also an organization" or "also a law" as the reason. The candidate must share a specific role/topic/name pattern.
- For bad_candidates, never return a likely alias, surname-only form, acronym, abbreviation,
  or expansion of the same entity. Example: "Mile Jedinak" -> "Jedinak" is not a bad candidate.
- For ORG bad_candidates, same broad type is not enough: university -> university, police -> police/law-enforcement, committee -> committee, health agency -> health agency, financial institution -> financial institution.
- For LAW bad_candidates, prefer a law with overlapping topic words or a confusingly similar title.
- For PERSON good_candidates, be very conservative. If the source entity is only one token
  (only a first name or only a surname), do not return expanded full names or first-name-only
  variants as good aliases unless the visible context explicitly and unambiguously proves the
  exact same person and the substituted sentence remains grammatical.

It is better to return an empty list than a low-quality candidate.
Keep item_id, entity_id, and entity unchanged.
Return only structured JSON.
"""


STOP_TOKENS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "for",
    "on",
    "in",
    "to",
    "with",
    "from",
    "by",
    "at",
    "act",
    "law",
    "department",
    "committee",
    "office",
    "bureau",
    "agency",
    "service",
    "services",
    "administration",
    "association",
    "foundation",
    "university",
    "college",
    "institute",
    "corporation",
    "corp",
    "inc",
    "company",
    "group",
    "national",
    "united",
    "states",
    "us",
    "u",
    "s",
    "state",
    "federal",
    "house",
    "senate",
    "council",
    "board",
    "commission",
    "ministry",
    "authority",
    "organization",
    "centre",
    "center",
    "society",
}


def load_mistral_client():
    try:
        from mistralai import Mistral
    except Exception:
        from mistralai.client import Mistral  # type: ignore

    load_env_file(Path("mvp/.env"))
    load_env_file(Path(".env"))
    api_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY or OPENAI_API_KEY")
    server_url = os.environ.get("MISTRAL_BASE_URL")
    if server_url:
        return Mistral(api_key=api_key, server_url=server_url)
    return Mistral(api_key=api_key)


def load_tags(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        provider = path.name.replace("entity_tags_top50k_", "").replace(".jsonl", "")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                entity_id = str(item["entity_id"])
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                item["provider"] = provider
                rows.append(item)
    return rows


def norm_literal(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def norm_alnum(text: str) -> str:
    return re.sub(r"[^\w]+", "", norm_literal(text))


def norm_article(text: str) -> str:
    value = norm_literal(text).replace("&", "and")
    value = re.sub(r"^(the|a|an)\s+", "", value)
    value = re.sub(r"('s|’s)$", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def is_same_or_trivial(entity: str, candidate: str) -> bool:
    return (
        norm_literal(entity) == norm_literal(candidate)
        or norm_alnum(entity) == norm_alnum(candidate)
        or norm_article(entity) == norm_article(candidate)
    )


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-я0-9]+", str(text or ""))


def is_ambiguous_person_good(entity: str, candidate: str, fine_type: str | None) -> bool:
    if fine_type != "PERSON":
        return False
    source_tokens = word_tokens(entity)
    candidate_tokens = word_tokens(candidate)
    if len(source_tokens) <= 1 and norm_alnum(entity) != norm_alnum(candidate):
        return True
    if len(candidate_tokens) <= 1 and len(source_tokens) <= 1 and norm_alnum(entity) != norm_alnum(candidate):
        return True
    return False


def is_possible_same_entity_surface(entity: str, candidate: str, fine_type: str | None) -> bool:
    entity_tokens = {token.casefold() for token in word_tokens(entity)}
    candidate_tokens = {token.casefold() for token in word_tokens(candidate)}
    if not entity_tokens or not candidate_tokens:
        return False
    if fine_type == "PERSON" and (
        candidate_tokens <= entity_tokens or entity_tokens <= candidate_tokens
    ):
        return True
    left = norm_article(entity)
    right = norm_article(candidate)
    return bool(left and right and (left in right or right in left) and token_overlap(entity, candidate) >= 0.5)


ORG_HEAD_TERMS = {
    "administration",
    "agency",
    "association",
    "authority",
    "bank",
    "board",
    "bureau",
    "college",
    "commission",
    "committee",
    "company",
    "corporation",
    "council",
    "department",
    "foundation",
    "institute",
    "ministry",
    "office",
    "school",
    "union",
    "university",
}


LEGAL_CONTEXT_PREFIX_RE = re.compile(
    r"^(section|subsection|paragraph|title|chapter|article|clause|part|funds?\s+of)\b",
    re.IGNORECASE,
)


def is_acronym_like(text: str) -> bool:
    stripped = re.sub(r"[^A-Za-z]", "", str(text or ""))
    return 2 <= len(stripped) <= 12 and stripped.upper() == stripped


def head_terms(text: str) -> set[str]:
    return {token.casefold() for token in word_tokens(text) if token.casefold() in ORG_HEAD_TERMS}


def is_suspicious_good_alias(entity: str, candidate: str, fine_type: str | None, relation: str | None) -> bool:
    if relation in {"abbreviation", "acronym"} or is_acronym_like(candidate):
        return False
    value = norm_literal(candidate)
    if LEGAL_CONTEXT_PREFIX_RE.search(value):
        return True
    entity_heads = head_terms(entity)
    candidate_heads = head_terms(candidate)
    if fine_type in {"ORG", "FAC"}:
        if entity_heads and candidate_heads and not (entity_heads & candidate_heads):
            return True
        if entity_heads & {"university", "college", "institute", "school"}:
            if len(word_tokens(candidate)) == 1 and not candidate_heads:
                return True
    return False


def content_tokens(text: str) -> set[str]:
    value = norm_literal(text).replace("u.s.", "us")
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value)
        if token not in STOP_TOKENS and len(token) > 1
    }


def token_overlap(left: str, right: str) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens and not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


ROLE_PATTERNS: list[tuple[str, str]] = [
    ("university", r"\b(university|college|school|institute|campus)\b"),
    ("police_law_enforcement", r"\b(police|sheriff|patrol|marshal|marshals|fbi|dea|atf|secret service|customs|border protection|law enforcement|public safety)\b"),
    ("health_agency", r"\b(health|medical|medicare|medicaid|disease|hospital|clinic|public health|human services)\b"),
    ("department_agency", r"\b(department|agency|administration|bureau|office|ministry|authority|commission)\b"),
    ("committee", r"\b(committee|subcommittee|caucus)\b"),
    ("court_judiciary", r"\b(court|judiciary|judicial|justice|attorney|prosecutor)\b"),
    ("financial", r"\b(bank|banking|finance|financial|mortgage|reserve|treasury|securities|credit)\b"),
    ("sports_event", r"\b(cup|league|championship|grand prix|open|tournament|football|basketball|athletic|olympic|games)\b"),
    ("law_act", r"\b(act|code|statute|law|amendment|constitution|title|public law|u\.s\.c|usc)\b"),
    ("nonprofit_association", r"\b(association|foundation|society|council|union)\b"),
]


SPECIFIC_ROLES = {
    "university",
    "police_law_enforcement",
    "health_agency",
    "committee",
    "court_judiciary",
    "financial",
    "sports_event",
    "law_act",
}


def role_labels(text: str) -> set[str]:
    value = norm_literal(text)
    roles = {name for name, pattern in ROLE_PATTERNS if re.search(pattern, value)}
    return roles or {"other"}


def hard_negative_score(entity: str, candidate: str, fine_type: str | None) -> float:
    overlap = token_overlap(entity, candidate)
    roles_left = role_labels(entity)
    roles_right = role_labels(candidate)
    if fine_type == "LAW":
        if "law_act" not in roles_left or "law_act" not in roles_right:
            return -0.5
        return overlap + (0.55 if overlap >= 0.10 else -0.20)
    specific_overlap = roles_left & roles_right & SPECIFIC_ROLES
    broad_overlap = roles_left & roles_right
    score = overlap
    if specific_overlap:
        score += 0.45
    elif broad_overlap and "other" not in broad_overlap:
        score += 0.18
    if fine_type == "ORG" and broad_overlap <= {"department_agency"}:
        score -= 0.2
    if roles_left == {"other"} and roles_right == {"other"} and overlap < 0.18:
        score -= 0.2
    return score


def is_close_hard_negative(entity: str, candidate: str, fine_type: str | None, candidate_pool: list[str]) -> bool:
    if is_same_or_trivial(entity, candidate):
        return False
    pool_keys = {norm_article(item) for item in candidate_pool}
    score = hard_negative_score(entity, candidate, fine_type)
    if norm_article(candidate) in pool_keys:
        return score >= 0.35
    return score >= 0.35


def pool_rank(entity: str, candidate: str, fine_type: str | None) -> tuple[float, str]:
    return (hard_negative_score(entity, candidate, fine_type), candidate.lower())



def context_for_agent(masked_text: str, mask_idx: int, entity: str, max_chars: int) -> str:
    text = replace_nth_mask(masked_text, mask_idx, entity)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = entity
    pos = text.find(marker)
    if pos < 0:
        return text[:max_chars]
    left = max(0, pos - max_chars // 2)
    right = min(len(text), left + max_chars)
    left = max(0, right - max_chars)
    return text[left:right].strip()


def aliasability_score(tag: dict[str, Any]) -> int:
    entity = str(tag.get("entity", ""))
    fine_type = str(tag.get("fine_type", ""))
    score = 0
    if fine_type in {"ORG", "GPE", "FAC", "LAW", "EVENT", "PRODUCT", "WORK_OF_ART", "NORP", "LOC"}:
        score += 50
    if fine_type == "PERSON":
        score += 5
    if len(entity) >= 18:
        score += 20
    if any(x in entity for x in ["U.S.", "United States", "University", "Department", "Committee", "Association", "Corporation", "Corp.", "Inc.", "Act", "Cup", "Open"]):
        score += 25
    if re.search(r"\b(the|The)\b", entity):
        score += 4
    return score


def choose_tags(tags: list[dict[str, Any]], limit: int, seed: int, offset: int = 0) -> list[dict[str, Any]]:
    proper = [t for t in tags if t.get("coarse_group") == "proper_name"]
    rng = random.Random(seed)
    rng.shuffle(proper)
    proper.sort(key=aliasability_score, reverse=True)
    if offset:
        proper = proper[offset:]
    return proper[:limit] if limit else proper


def prepare_items(
    tags: list[dict[str, Any]],
    conn: sqlite3.Connection,
    max_context_chars: int,
    seed: int,
    pool_by_fine_type: dict[str, list[str]],
    pool_size: int,
    mentions_per_entity: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for tag in tags:
        mentions = sample_mentions(conn, str(tag["entity_id"]), rng, mentions_per_entity)
        if not mentions:
            continue
        fine_type = str(tag.get("fine_type"))
        pool_candidates = [
            candidate
            for candidate in pool_by_fine_type.get(fine_type, [])
            if not is_same_or_trivial(tag["entity"], candidate)
            and hard_negative_score(tag["entity"], candidate, fine_type) >= 0.18
        ]
        pool_candidates.sort(
            key=lambda candidate: pool_rank(tag["entity"], candidate, fine_type),
            reverse=True,
        )
        head = pool_candidates[: max(pool_size * 3, pool_size)]
        candidate_pool = rng.sample(head, min(pool_size, len(head))) if head else []
        for mention in mentions:
            mask_idx = int(mention["mask_idx"])
            full_context = context_for_agent(mention["masked_text"], mask_idx, tag["entity"], max_context_chars)
            mention_id = mention.get("mention_id")
            item_id = f"{tag['entity_id']}:{mention_id if mention_id is not None else mask_idx}"
            items.append(
                {
                    "item_id": item_id,
                    "entity_id": str(tag["entity_id"]),
                    "entity": tag["entity"],
                    "fine_type": fine_type,
                    "coarse_group": tag.get("coarse_group"),
                    "provider": tag.get("provider"),
                    "dataset": f"{mention['dataset_name']}/{mention['dataset_split']}",
                    "dataset_name": mention.get("dataset_name"),
                    "dataset_split": mention.get("dataset_split"),
                    "dataset_id": mention.get("dataset_id"),
                    "text_id": mention.get("text_id"),
                    "mention_id": mention.get("mention_id"),
                    "row_idx": mention.get("row_idx"),
                    "source_id": mention.get("source_id"),
                    "domain": mention["domain"],
                    "mask_idx": mask_idx,
                    "masked_text": mention["masked_text"],
                    "full_context": full_context,
                    "role_signature": sorted(role_labels(tag["entity"])),
                    "candidate_pool": candidate_pool,
                }
            )
    return items


def compact_for_prompt(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item["item_id"],
        "entity_id": item["entity_id"],
        "entity": item["entity"],
        "fine_type": item.get("fine_type"),
        "role_signature": item.get("role_signature"),
        "dataset": item.get("dataset"),
        "domain": item.get("domain"),
        "candidate_pool": item.get("candidate_pool", []),
        "full_context": item["full_context"],
    }


def call_alias_agent(client, model: str, batch: list[dict[str, Any]], max_tokens: int) -> AliasBatch:
    payload = [compact_for_prompt(item) for item in batch]
    response = client.chat.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Generate safe same-entity aliases and hard negative candidates for these named entities. "
                    "Keep entity_id and entity unchanged. Return empty candidates when unsure.\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ],
        response_format=AliasBatch,
        temperature=0,
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        return parsed
    return AliasBatch.model_validate_json(message.content)


def call_with_retries(
    client,
    model: str,
    batch: list[dict[str, Any]],
    max_tokens: int,
    retries: int,
    retry_sleep: float,
    batch_idx: int,
) -> AliasBatch:
    attempt = 0
    last_error: Exception | None = None
    while True:
        try:
            return call_alias_agent(client, model, batch, max_tokens)
        except Exception as exc:
            last_error = exc
            attempt += 1
            if attempt > retries:
                raise last_error
            delay = retry_sleep * min(2 ** (attempt - 1), 8)
            print(f"batch={batch_idx} retry={attempt}/{retries} sleep={delay:.1f}s error={type(exc).__name__}: {exc}", flush=True)
            time.sleep(delay)


def make_pair(
    item: dict[str, Any],
    candidate: GoodCandidate | BadCandidate,
    pair_id: str,
    pair_context_chars: int,
    expected_score: float,
    candidate_kind: str,
) -> dict[str, Any]:
    original_context, candidate_context = fill_contexts(
        item["masked_text"],
        int(item["mask_idx"]),
        item["entity"],
        candidate.candidate,
        pair_context_chars,
    )
    return {
        "pair_id": pair_id,
        "item_id": item.get("item_id"),
        "branch": "proper_name_alias_agent",
        "candidate_kind": candidate_kind,
        "expected_score": expected_score,
        "entity_id": item["entity_id"],
        "entity": item["entity"],
        "candidate": candidate.candidate,
        "alias_relation": candidate.relation,
        "alias_confidence": candidate.confidence,
        "alias_rationale": getattr(candidate, "rationale", None),
        "negative_why_plausible": getattr(candidate, "why_plausible", None),
        "negative_why_wrong": getattr(candidate, "why_wrong", None),
        "fine_type": item.get("fine_type"),
        "coarse_group": "proper_name",
        "context_policy": "full_context",
        "provider": item.get("provider"),
        "dataset": item.get("dataset"),
        "dataset_name": item.get("dataset_name"),
        "dataset_split": item.get("dataset_split"),
        "dataset_id": item.get("dataset_id"),
        "text_id": item.get("text_id"),
        "mention_id": item.get("mention_id"),
        "row_idx": item.get("row_idx"),
        "source_id": item.get("source_id"),
        "domain": item.get("domain"),
        "mask_idx": item.get("mask_idx"),
        "short_2w": "",
        "original_context": original_context,
        "candidate_context": candidate_context,
    }


def existing_item_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                ids.add(str(row.get("item_id") or row.get("entity_id")))
    return ids


def build_pool_by_fine_type(tags: list[dict[str, Any]]) -> dict[str, list[str]]:
    pools: dict[str, set[str]] = {}
    for tag in tags:
        if tag.get("coarse_group") != "proper_name":
            continue
        fine_type = str(tag.get("fine_type", "UNKNOWN"))
        surface = str(tag.get("entity", "")).strip()
        if len(surface) < 2:
            continue
        pools.setdefault(fine_type, set()).add(surface)
    return {key: sorted(values, key=lambda text: (-len(text), text.lower())) for key, values in pools.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-files", type=Path, nargs="+", required=True)
    parser.add_argument("--db", type=Path, default=Path("data/entity_inventory.sqlite"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--raw-out", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--pair-context-chars", type=int, default=1800)
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--mentions-per-entity", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--sleep", type=float, default=0.7)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    load_env_file(Path("mvp/.env"))
    load_env_file(Path(".env"))
    model = args.model or os.environ.get("MISTRAL_MODEL") or "mistral-small-latest"

    all_tags = load_tags(args.tag_files)
    pool_by_fine_type = build_pool_by_fine_type(all_tags)
    tags = choose_tags(all_tags, args.limit, args.seed, args.offset)

    conn = connect_ro(args.db)
    items = prepare_items(
        tags,
        conn,
        args.max_context_chars,
        args.seed,
        pool_by_fine_type,
        args.pool_size,
        args.mentions_per_entity,
    )
    if args.resume:
        done = existing_item_keys(args.raw_out or args.out)
        before = len(items)
        items = [item for item in items if str(item.get("item_id") or item["entity_id"]) not in done]
        print(
            f"resume=true existing_items={len(done)} remaining_items={len(items)} "
            f"skipped_items={before - len(items)}",
            flush=True,
        )
    print(f"items={len(items)} model={model}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_out or args.out.with_suffix(".raw.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    client = load_mistral_client()
    out_mode = "a" if args.resume and args.out.exists() else "w"
    raw_mode = "a" if args.resume and raw_path.exists() else "w"

    pair_count = 0
    entity_with_candidates = 0
    rejected_good_trivial = 0
    rejected_good_ambiguous_person = 0
    rejected_good_suspicious = 0
    rejected_bad_trivial = 0
    rejected_bad_weak = 0
    with args.out.open(out_mode, encoding="utf-8") as pair_fh, raw_path.open(raw_mode, encoding="utf-8") as raw_fh:
        batches = [items[i : i + args.batch_size] for i in range(0, len(items), args.batch_size)]
        for batch_idx, batch in enumerate(batches, start=1):
            started = time.time()
            result = call_with_retries(
                client,
                model,
                batch,
                args.max_tokens,
                args.retries,
                args.retry_sleep,
                batch_idx,
            )
            by_id = {item.item_id: item for item in result.items}
            for source in batch:
                output = by_id.get(str(source["item_id"])) or AliasItem(
                    item_id=str(source["item_id"]),
                    entity_id=str(source["entity_id"]),
                    entity=source["entity"],
                    good_candidates=[],
                    bad_candidates=[],
                )
                raw_row = output.model_dump()
                raw_row.update(
                    {
                        "item_id": source.get("item_id"),
                        "dataset": source.get("dataset"),
                        "dataset_name": source.get("dataset_name"),
                        "dataset_split": source.get("dataset_split"),
                        "dataset_id": source.get("dataset_id"),
                        "text_id": source.get("text_id"),
                        "mention_id": source.get("mention_id"),
                        "row_idx": source.get("row_idx"),
                        "source_id": source.get("source_id"),
                        "domain": source.get("domain"),
                        "mask_idx": source.get("mask_idx"),
                    }
                )
                raw_fh.write(json.dumps(raw_row, ensure_ascii=False) + "\n")
                raw_good_candidates = output.good_candidates
                raw_bad_candidates = output.bad_candidates
                kept_good = []
                kept_bad = []
                seen: set[str] = set()
                for candidate in raw_good_candidates:
                    value = candidate.candidate.strip()
                    key = norm_alnum(value)
                    if not value or key in seen or is_same_or_trivial(source["entity"], value):
                        rejected_good_trivial += 1
                        continue
                    if is_ambiguous_person_good(source["entity"], value, str(source.get("fine_type"))):
                        rejected_good_ambiguous_person += 1
                        continue
                    if is_suspicious_good_alias(
                        source["entity"],
                        value,
                        str(source.get("fine_type")),
                        str(candidate.relation),
                    ):
                        rejected_good_suspicious += 1
                        continue
                    seen.add(key)
                    kept_good.append(candidate)
                for candidate in raw_bad_candidates:
                    value = candidate.candidate.strip()
                    key = norm_alnum(value)
                    if not value or key in seen or is_same_or_trivial(source["entity"], value):
                        rejected_bad_trivial += 1
                        continue
                    if not is_close_hard_negative(
                        source["entity"],
                        value,
                        str(source.get("fine_type")),
                        list(source.get("candidate_pool", [])),
                    ):
                        rejected_bad_weak += 1
                        continue
                    if is_possible_same_entity_surface(source["entity"], value, str(source.get("fine_type"))):
                        rejected_bad_weak += 1
                        continue
                    seen.add(key)
                    kept_bad.append(candidate)
                if kept_good or kept_bad:
                    entity_with_candidates += 1
                for candidate in kept_good:
                    pair_count += 1
                    pair = make_pair(
                        source,
                        candidate,
                        f"pa{pair_count:06d}",
                        args.pair_context_chars,
                        1.0,
                        "proper_agent_alias",
                    )
                    pair_fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                for candidate in kept_bad:
                    pair_count += 1
                    pair = make_pair(
                        source,
                        candidate,
                        f"pa{pair_count:06d}",
                        args.pair_context_chars,
                        0.0,
                        "proper_agent_hard_negative",
                    )
                    pair_fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
            pair_fh.flush()
            raw_fh.flush()
            print(
                f"batch={batch_idx}/{len(batches)} entities={len(batch)} "
                f"pairs={pair_count} entity_with_candidates={entity_with_candidates} "
                f"rejected_good_trivial={rejected_good_trivial} "
                f"rejected_good_ambiguous_person={rejected_good_ambiguous_person} "
                f"rejected_good_suspicious={rejected_good_suspicious} "
                f"rejected_bad_trivial={rejected_bad_trivial} rejected_bad_weak={rejected_bad_weak} "
                f"seconds={time.time() - started:.2f}",
                flush=True,
            )
            if args.sleep:
                time.sleep(args.sleep)

    print(f"wrote_pairs={pair_count} out={args.out}", flush=True)
    print(f"wrote_raw={raw_path}", flush=True)
    print(
        f"rejected_good_trivial={rejected_good_trivial} "
        f"rejected_good_ambiguous_person={rejected_good_ambiguous_person} "
        f"rejected_good_suspicious={rejected_good_suspicious} "
        f"rejected_bad_trivial={rejected_bad_trivial} rejected_bad_weak={rejected_bad_weak}",
        flush=True,
    )


if __name__ == "__main__":
    main()
