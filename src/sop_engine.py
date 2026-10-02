"""Generic deterministic SOP evaluation engine.

Evaluates declarative SOP rules against computed facts and requested activity tags.
Implements three-valued logic (True, False, None/Unevaluable) for robust safety enforcement.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel

from src.sop_schema import (
    SOP,
    ConditionNode,
    LeafCondition,
    CompoundCondition,
    SeverityLevel,
)

SOPStatus = str  # "matched" | "not_matched" | "unevaluable" | "not_applicable"


class SOPResult(BaseModel):
    """Result of evaluating an SOP against a specific set of facts and activity tags."""
    sop_id: str
    status: str  # "matched", "not_matched", "unevaluable", "not_applicable"
    effective_severity: SeverityLevel
    rendered_advice: str
    facts_used: Dict[str, Any]
    tier: Optional[str] = None
    override: bool = False
    priority: int = 100
    category: str = "general"
    match_type: str = "numeric"


def _format_fact_value(val: Any) -> str:
    """Format fact value for placeholder substitution (round floats to 1 decimal place)."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        # E.g. 28.5
        return f"{val:.1f}"
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


def render_advice_text(template: str, facts: Dict[str, Any]) -> str:
    """Render advice template by safely substituting {fact_name} tokens from facts."""
    if not template:
        return ""

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in facts:
            return _format_fact_value(facts[key])
        return match.group(0)

    return re.sub(r"\{([a-zA-Z0-9_]+)\}", _replace, template)


def evaluate_leaf(
    leaf: LeafCondition,
    facts: Dict[str, Any],
) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Evaluate single leaf condition.

    Returns:
        (True/False/None, facts_used) where None indicates unevaluable due to None/missing fact.
    """
    fact_val = facts.get(leaf.fact)
    facts_used = {leaf.fact: fact_val}

    if fact_val is None:
        return None, facts_used

    op = leaf.op
    target = leaf.value

    try:
        if op == ">":
            return bool(fact_val > target), facts_used
        elif op == ">=":
            return bool(fact_val >= target), facts_used
        elif op == "<":
            return bool(fact_val < target), facts_used
        elif op == "<=":
            return bool(fact_val <= target), facts_used
        elif op == "==":
            return bool(fact_val == target), facts_used
        elif op == "!=":
            return bool(fact_val != target), facts_used
        elif op == "in":
            return bool(fact_val in target), facts_used
        elif op == "intersects":
            val_set = set(fact_val) if isinstance(fact_val, (list, set, tuple)) else {fact_val}
            target_set = set(target) if isinstance(target, (list, set, tuple)) else {target}
            return bool(val_set.intersection(target_set)), facts_used
        else:
            return False, facts_used
    except TypeError:
        return False, facts_used


def evaluate_condition_node(
    node: ConditionNode,
    facts: Dict[str, Any],
) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Evaluate condition node with three-valued logic.

    Logic rules:
    - any_of: If any child is True -> True. If all are False -> False. Otherwise (mix of False and None) -> None.
    - all_of: If any child is False -> False. If all are True -> True. Otherwise (mix of True and None) -> None.
    """
    if isinstance(node, LeafCondition):
        return evaluate_leaf(node, facts)

    facts_used: Dict[str, Any] = {}

    if node.any_of is not None:
        has_none = False
        for child in node.any_of:
            res, child_facts = evaluate_condition_node(child, facts)
            facts_used.update(child_facts)
            if res is True:
                return True, facts_used
            if res is None:
                has_none = True

        return (None if has_none else False), facts_used

    if node.all_of is not None:
        has_none = False
        for child in node.all_of:
            res, child_facts = evaluate_condition_node(child, facts)
            facts_used.update(child_facts)
            if res is False:
                return False, facts_used
            if res is None:
                has_none = True

        return (None if has_none else True), facts_used

    return False, facts_used


def is_applicable(sop: SOP, activity_tags: List[str]) -> bool:
    """Check if SOP applies to user's activity tags."""
    if "any" in [tag.lower() for tag in sop.applies_to]:
        return True

    user_tags = {tag.strip().lower() for tag in activity_tags}
    sop_tags = {tag.strip().lower() for tag in sop.applies_to}
    return bool(user_tags.intersection(sop_tags))


def extract_thresholds(sop: SOP) -> Set[float | int]:
    """Extract all numeric thresholds appearing in conditions and advice text."""
    thresholds: Set[float | int] = set()

    def _collect_from_cond(cond: Optional[ConditionNode]) -> None:
        if cond is None:
            return
        if isinstance(cond, LeafCondition):
            if isinstance(cond.value, (int, float)):
                thresholds.add(cond.value)
            elif isinstance(cond.value, (list, set, tuple)):
                for item in cond.value:
                    if isinstance(item, (int, float)):
                        thresholds.add(item)
        elif isinstance(cond, CompoundCondition):
            children = cond.all_of or cond.any_of or []
            for child in children:
                _collect_from_cond(child)

    _collect_from_cond(sop.condition)

    if sop.rubric:
        for tier in sop.rubric:
            _collect_from_cond(tier.condition)

    # Also capture explicit static numbers written in advice strings (ignoring placeholders)
    # E.g. "Take a break every 30 minutes" -> 30
    advice_texts = [sop.advice]
    if sop.rubric:
        advice_texts.extend(t.advice for t in sop.rubric)

    for text in advice_texts:
        clean_text = re.sub(r"\{[a-zA-Z0-9_]+\}", "", text)
        for num_str in re.findall(r"\b\d+(?:\.\d+)?\b", clean_text):
            val = float(num_str) if "." in num_str else int(num_str)
            thresholds.add(val)

    return thresholds


def evaluate_sop(
    sop: SOP,
    facts: Dict[str, Any],
    activity_tags: List[str],
) -> SOPResult:
    """Evaluate a single SOP (numeric, composite, or semantic).

    Note: Clear SOPs should be evaluated through evaluate_all to incorporate category baseline.
    """
    if not is_applicable(sop, activity_tags):
        return SOPResult(
            sop_id=sop.id,
            status="not_applicable",
            effective_severity=sop.severity,
            rendered_advice="",
            facts_used={},
            override=sop.override,
            priority=sop.priority,
            category=sop.category,
            match_type=sop.match_type,
        )

    # 1. Numeric or Composite SOP
    if sop.match_type in ("numeric", "composite"):
        if sop.condition is None:
            status = "not_matched"
            facts_used = {}
        else:
            cond_res, facts_used = evaluate_condition_node(sop.condition, facts)
            if cond_res is True:
                status = "matched"
            elif cond_res is None:
                status = "unevaluable"
            else:
                status = "not_matched"

        rendered_advice = render_advice_text(sop.advice, facts) if status == "matched" else ""
        return SOPResult(
            sop_id=sop.id,
            status=status,
            effective_severity=sop.severity,
            rendered_advice=rendered_advice,
            facts_used=facts_used,
            override=sop.override,
            priority=sop.priority,
            category=sop.category,
            match_type=sop.match_type,
        )

    # 2. Semantic SOP
    if sop.match_type == "semantic":
        all_facts_used: Dict[str, Any] = {}
        has_unevaluable_tier = False

        if sop.rubric:
            for tier in sop.rubric:
                cond_res, tier_facts = evaluate_condition_node(tier.condition, facts)
                all_facts_used.update(tier_facts)

                if cond_res is True:
                    # First winning tier
                    rendered_advice = render_advice_text(tier.advice, facts)
                    return SOPResult(
                        sop_id=sop.id,
                        status="matched",
                        effective_severity=tier.severity,
                        rendered_advice=rendered_advice,
                        facts_used=all_facts_used,
                        tier=tier.name,
                        override=sop.override,
                        priority=sop.priority,
                        category=sop.category,
                        match_type=sop.match_type,
                    )
                if cond_res is None:
                    has_unevaluable_tier = True

        status = "unevaluable" if has_unevaluable_tier else "not_matched"
        return SOPResult(
            sop_id=sop.id,
            status=status,
            effective_severity=sop.severity,
            rendered_advice="",
            facts_used=all_facts_used,
            override=sop.override,
            priority=sop.priority,
            category=sop.category,
            match_type=sop.match_type,
        )

    # 3. Clear SOP placeholder if called standalone
    return SOPResult(
        sop_id=sop.id,
        status="not_matched",
        effective_severity=sop.severity,
        rendered_advice="",
        facts_used={},
        override=sop.override,
        priority=sop.priority,
        category=sop.category,
        match_type=sop.match_type,
    )


def evaluate_all(
    sops: List[SOP],
    facts: Dict[str, Any],
    activity_tags: List[str],
) -> Tuple[List[SOPResult], List[str], List[str]]:
    """Evaluate all SOPs against computed facts and activity tags.

    Enforces category-wide and general-wide evaluability rules for 'clear' baseline SOPs.

    Returns:
        Tuple of (all_results, matched_ids, unevaluable_ids)
    """
    results: List[SOPResult] = []

    # Pass 1: Evaluate all non-clear SOPs
    non_clear_sops = [s for s in sops if s.match_type != "clear"]
    clear_sops = [s for s in sops if s.match_type == "clear"]

    for sop in non_clear_sops:
        res = evaluate_sop(sop, facts, activity_tags)
        results.append(res)
    applicable_hazard_statuses = [
        r.status for r in results if r.status != "not_applicable" and r.match_type != "clear"
    ]

    # Pass 2: Evaluate clear SOPs
    for sop in clear_sops:
        if not is_applicable(sop, activity_tags):
            results.append(
                SOPResult(
                    sop_id=sop.id,
                    status="not_applicable",
                    effective_severity=sop.severity,
                    rendered_advice="",
                    facts_used={},
                    override=sop.override,
                    priority=sop.priority,
                    category=sop.category,
                    match_type=sop.match_type,
                )
            )
            continue

        # A clear SOP considers EVERY non-clear SOP applicable to the activity tags (any category)
        if any(st == "unevaluable" for st in applicable_hazard_statuses):
            # A clear SOP never fires when any applicable hazard SOP is unevaluable
            res_status = "unevaluable"
        elif any(st == "matched" for st in applicable_hazard_statuses):
            res_status = "not_matched"
        else:
            # All applicable hazards were evaluable and none matched
            res_status = "matched"

        rendered_advice = render_advice_text(sop.advice, facts) if res_status == "matched" else ""
        results.append(
            SOPResult(
                sop_id=sop.id,
                status=res_status,
                effective_severity=sop.severity,
                rendered_advice=rendered_advice,
                facts_used={},
                override=sop.override,
                priority=sop.priority,
                category=sop.category,
                match_type=sop.match_type,
            )
        )

    matched_ids = [r.sop_id for r in results if r.status == "matched"]
    unevaluable_ids = [r.sop_id for r in results if r.status == "unevaluable"]

    return results, matched_ids, unevaluable_ids
