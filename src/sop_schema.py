"""SOP (Standard Operating Procedure) schemas and validation.

Defines Pydantic v2 schemas for declarative YAML policies, condition trees,
tiered semantic rubrics, and strict configuration loading rules.
"""

from __future__ import annotations

import re
from typing import Any, List, Literal, Optional, Set, Union
from pydantic import BaseModel, Field, model_validator

SeverityLevel = Literal["info", "low", "moderate", "high", "critical"]
SEVERITY_WEIGHTS: dict[SeverityLevel, int] = {
    "info": 1,
    "low": 2,
    "moderate": 3,
    "high": 4,
    "critical": 5,
}

MatchType = Literal["numeric", "composite", "semantic", "clear"]
OperatorType = Literal[">", ">=", "<", "<=", "==", "!=", "in", "intersects"]


class LeafCondition(BaseModel):
    """Leaf condition evaluating a single fact against an operator and target value."""
    fact: str
    op: OperatorType
    value: Any


class CompoundCondition(BaseModel):
    """Compound condition combining child conditions using all_of or any_of logic."""
    all_of: Optional[List[Union[CompoundCondition, LeafCondition]]] = None
    any_of: Optional[List[Union[CompoundCondition, LeafCondition]]] = None

    @model_validator(mode="after")
    def validate_node(self) -> CompoundCondition:
        if self.all_of is None and self.any_of is None:
            raise ValueError("Compound condition must specify either 'all_of' or 'any_of'.")
        if self.all_of is not None and self.any_of is not None:
            raise ValueError("Compound condition cannot specify both 'all_of' and 'any_of'.")
        return self


ConditionNode = Union[CompoundCondition, LeafCondition]


class RubricTier(BaseModel):
    """Tier in a deterministic semantic rubric evaluated in priority order."""
    name: str
    condition: ConditionNode
    severity: SeverityLevel
    advice: str


class SOP(BaseModel):
    """Standard Operating Procedure specification."""
    id: str = Field(description="Unique identifier e.g. EXR-001")
    title: str
    category: str
    severity: SeverityLevel
    priority: int = Field(ge=0, description="Evaluation rank; lower integer has higher priority")
    applies_to: List[str] = Field(description="Activity/scenario tags or ['any']")
    match_type: MatchType
    override: bool = False
    condition: Optional[ConditionNode] = None
    advice: str = ""
    rationale: str = ""
    intent_description: Optional[str] = None
    rubric: Optional[List[RubricTier]] = None

    @model_validator(mode="after")
    def validate_sop_type_contracts(self) -> SOP:
        """Validate consistency according to match_type."""
        if self.match_type in ("numeric", "composite"):
            if self.condition is None:
                raise ValueError(f"SOP '{self.id}' with match_type '{self.match_type}' requires a 'condition'.")
        elif self.match_type == "semantic":
            if not self.rubric:
                raise ValueError(f"Semantic SOP '{self.id}' requires an ordered 'rubric' list.")
            if not self.intent_description:
                raise ValueError(f"Semantic SOP '{self.id}' requires an 'intent_description'.")
        elif self.match_type == "clear":
            # Clear SOPs must not define conflicting conditions
            if self.condition is not None:
                raise ValueError(f"Clear SOP '{self.id}' must not define a 'condition' (evaluated by category baseline).")
        return self

    def extract_placeholders(self) -> Set[str]:
        """Extract all {fact_name} tokens from advice and rubric advice strings."""
        placeholders: Set[str] = set()
        pattern = re.compile(r"\{([a-zA-Z0-9_]+)\}")
        if self.advice:
            for match in pattern.finditer(self.advice):
                placeholders.add(match.group(1))
        if self.rubric:
            for tier in self.rubric:
                for match in pattern.finditer(tier.advice):
                    placeholders.add(match.group(1))
        return placeholders

    def extract_referenced_facts(self) -> Set[str]:
        """Extract all fact names referenced in conditions and placeholders."""
        facts = self.extract_placeholders()

        def _collect(cond: ConditionNode) -> None:
            if isinstance(cond, LeafCondition):
                facts.add(cond.fact)
            elif isinstance(cond, CompoundCondition):
                children = cond.all_of or cond.any_of or []
                for child in children:
                    _collect(child)

        if self.condition:
            _collect(self.condition)

        if self.rubric:
            for tier in self.rubric:
                _collect(tier.condition)

        return facts
