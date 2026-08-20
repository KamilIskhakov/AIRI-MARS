from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EntityCoarseGroup = Literal[
    "proper_name",
    "numeric",
    "common_entity",
    "domain_term",
    "ambiguous",
    "junk",
]

EntityFineType = Literal[
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "NORP",
    "EVENT",
    "LAW",
    "PRODUCT",
    "WORK_OF_ART",
    "DATE",
    "TIME",
    "CARDINAL",
    "ORDINAL",
    "MONEY",
    "PERCENT",
    "QUANTITY",
    "COMMON_NOUN",
    "DOMAIN_TERM",
    "OTHER",
    "UNKNOWN",
    "JUNK",
]

ContextPolicy = Literal[
    "short_window",
    "full_context",
    "no_context_embedding",
    "agent_review",
    "drop",
]


class EntityTag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(description="Stable id from the input inventory item.")
    entity: str = Field(description="Surface form of the entity exactly as provided.")
    coarse_group: EntityCoarseGroup
    fine_type: EntityFineType
    context_policy: ContextPolicy
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=240)


class EntityTagBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[EntityTag]
