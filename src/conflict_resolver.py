"""Deterministic conflict resolver for matched SOP advisories.

POLICY AND RANKING ORDER:
When multiple SOPs match the current meteorological facts and user activity, they are
ranked by the following deterministic precedence rules:
1. Override Flag: An SOP with 'override: true' strictly takes precedence over any SOP with 'override: false'.
2. Severity Level: Higher hazard severity outranks lower severity:
   critical (5) > high (4) > moderate (3) > low (2) > info (1).
3. Priority Integer: Lower priority integer represents higher operational rank (e.g. priority 1 beats priority 2).
4. SOP Identifier: Lexicographical tie-breaking on unique 'id' (e.g. 'EXR-001' beats 'EXR-002').

The top-ranked SOP is designated as 'primary', remaining matches are designated as 'also_applies',
and an explainable 'reason' string is generated to document the decision in the audit log.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel

from src.sop_engine import SOPResult
from src.sop_schema import SEVERITY_WEIGHTS


class ConflictResolution(BaseModel):
    """Structured outcome of conflict resolution across matched SOPs."""
    primary: Optional[SOPResult] = None
    also_applies: List[SOPResult] = []
    reason: str = ""


def _sort_key(res: SOPResult) -> tuple:
    """Sort key for SOP ranking: override (desc), severity (desc), priority (asc), id (asc)."""
    # Python sorts ascending:
    # 0 for override=True, 1 for override=False
    override_rank = 0 if res.override else 1
    # -weight for higher severity first
    sev_rank = -SEVERITY_WEIGHTS.get(res.effective_severity, 1)
    # lower integer priority first
    priority_rank = res.priority
    # alphabetical id
    id_rank = res.sop_id

    return (override_rank, sev_rank, priority_rank, id_rank)


def _build_reason(primary: SOPResult, runner_up: Optional[SOPResult]) -> str:
    """Generate human-readable justification for why primary won."""
    if runner_up is None:
        return f"SOP '{primary.sop_id}' was selected as the sole matching advisory."

    if primary.override and not runner_up.override:
        return (
            f"SOP '{primary.sop_id}' was selected because it specifies an operational override "
            f"over '{runner_up.sop_id}'."
        )

    p_weight = SEVERITY_WEIGHTS.get(primary.effective_severity, 1)
    r_weight = SEVERITY_WEIGHTS.get(runner_up.effective_severity, 1)
    if p_weight > r_weight:
        return (
            f"SOP '{primary.sop_id}' was selected due to higher severity "
            f"('{primary.effective_severity}' vs '{runner_up.effective_severity}' for '{runner_up.sop_id}')."
        )

    if primary.priority < runner_up.priority:
        return (
            f"SOP '{primary.sop_id}' was selected due to higher operational priority "
            f"(rank {primary.priority} vs {runner_up.priority} for '{runner_up.sop_id}')."
        )

    return (
        f"SOP '{primary.sop_id}' was selected via deterministic tie-breaking on ID "
        f"over '{runner_up.sop_id}'."
    )


def resolve(matched_results: List[SOPResult]) -> ConflictResolution:
    """Resolve matched SOPs into a single primary advisory and secondary advisories.

    Args:
        matched_results: List of SOPResults with status == 'matched'.

    Returns:
        ConflictResolution object containing primary, also_applies, and rationale reason.
    """
    if not matched_results:
        return ConflictResolution(
            primary=None,
            also_applies=[],
            reason="No SOPs matched current conditions.",
        )

    # Sort candidates
    sorted_matches = sorted(matched_results, key=_sort_key)
    primary = sorted_matches[0]
    also_applies = sorted_matches[1:]
    runner_up = also_applies[0] if also_applies else None

    reason = _build_reason(primary, runner_up)

    return ConflictResolution(
        primary=primary,
        also_applies=also_applies,
        reason=reason,
    )
