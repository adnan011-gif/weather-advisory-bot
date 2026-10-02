"""SOP loader and dynamic activity vocabulary generator.

Loads all SOP YAML files from a directory, validates against facts.yaml registry,
enforces duplicate ID protection, and derives activity/scenario vocabularies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml
from pydantic import ValidationError

from src.facts_registry import FactsRegistry
from src.sop_schema import SOP, LeafCondition, CompoundCondition, ConditionNode


class SOPLoadError(Exception):
    """Raised when an SOP file is malformed, invalid, or violates domain constraints."""
    pass


def _validate_leaf_condition(
    leaf: LeafCondition,
    registered_facts: Set[str],
    filename: str,
) -> None:
    """Validate that leaf condition references a known fact and valid operator."""
    if leaf.fact not in registered_facts:
        raise SOPLoadError(
            f"File '{filename}', field 'condition': referenced fact '{leaf.fact}' "
            f"is not defined in facts.yaml. Known facts: {sorted(list(registered_facts))}"
        )

    if leaf.op in (">", ">=", "<", "<="):
        if not isinstance(leaf.value, (int, float)):
            raise SOPLoadError(
                f"File '{filename}', field 'condition': numeric operator '{leaf.op}' "
                f"requires a numeric value, got {type(leaf.value).__name__} ({leaf.value})."
            )
    elif leaf.op == "intersects":
        if not isinstance(leaf.value, (list, set, tuple)):
            raise SOPLoadError(
                f"File '{filename}', field 'condition': operator 'intersects' "
                f"requires a collection value, got {type(leaf.value).__name__}."
            )
    elif leaf.op == "in":
        if not isinstance(leaf.value, (list, set, tuple)):
            raise SOPLoadError(
                f"File '{filename}', field 'condition': operator 'in' "
                f"requires a collection value, got {type(leaf.value).__name__}."
            )


def _validate_condition_tree(
    cond: ConditionNode,
    registered_facts: Set[str],
    filename: str,
) -> None:
    """Recursively validate conditions in condition tree."""
    if isinstance(cond, LeafCondition):
        _validate_leaf_condition(cond, registered_facts, filename)
    elif isinstance(cond, CompoundCondition):
        children = cond.all_of or cond.any_of or []
        for child in children:
            _validate_condition_tree(child, registered_facts, filename)


def load_sops(
    sop_dir: Path | str,
    facts_registry: Optional[FactsRegistry] = None,
) -> List[SOP]:
    """Load all SOPs from YAML files in the given directory.

    Args:
        sop_dir: Directory containing individual SOP YAML files.
        facts_registry: Optional FactsRegistry for validating fact existence.

    Returns:
        List of validated SOP objects.

    Raises:
        SOPLoadError: If any file cannot be read, parsed, or violates schema/domain rules.
    """
    directory = Path(sop_dir)
    if not directory.exists() or not directory.is_dir():
        raise SOPLoadError(f"SOP directory not found: {directory}")

    registry = facts_registry or FactsRegistry()
    registered_facts = {f.name for f in registry.config.facts}

    loaded_sops: List[SOP] = []
    seen_ids: Dict[str, str] = {}  # id -> filename

    # Sort files for deterministic loading order
    yaml_files = sorted(list(directory.glob("*.yaml")) + list(directory.glob("*.yml")))

    for filepath in yaml_files:
        filename = filepath.name
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        except Exception as err:
            raise SOPLoadError(f"File '{filename}': failed to parse YAML: {err}") from err

        if not isinstance(raw_data, dict):
            raise SOPLoadError(f"File '{filename}': expected a YAML dictionary, got {type(raw_data).__name__}.")

        try:
            sop = SOP.model_validate(raw_data)
        except ValidationError as err:
            field_errors = []
            for e in err.errors():
                loc = ".".join(str(p) for p in e["loc"])
                field_errors.append(f"field '{loc}': {e['msg']}")
            raise SOPLoadError(f"File '{filename}', {'; '.join(field_errors)}") from err
        except ValueError as err:
            raise SOPLoadError(f"File '{filename}': {err}") from err

        # Check duplicate ID
        if sop.id in seen_ids:
            raise SOPLoadError(
                f"File '{filename}', field 'id': duplicate SOP ID '{sop.id}' already declared in '{seen_ids[sop.id]}'."
            )
        seen_ids[sop.id] = filename

        # Validate that all referenced facts exist in facts.yaml
        referenced_facts = sop.extract_referenced_facts()
        unknown_facts = referenced_facts - registered_facts
        if unknown_facts:
            raise SOPLoadError(
                f"File '{filename}': references unknown fact(s) {sorted(list(unknown_facts))}. "
                f"Must be defined in facts.yaml."
            )

        # Validate condition operators and values
        if sop.condition:
            _validate_condition_tree(sop.condition, registered_facts, filename)

        if sop.rubric:
            for tier in sop.rubric:
                _validate_condition_tree(tier.condition, registered_facts, filename)

        loaded_sops.append(sop)

    return loaded_sops


def derive_vocabulary(sops: List[SOP]) -> Dict[str, Dict[str, str]]:
    """Derive dynamic activity/scenario vocabulary from applies_to tags of loaded SOPs.

    Args:
        sops: Collection of loaded SOPs.

    Returns:
        Mapping of activity tag -> {'title': str, 'intent_description': str}.
    """
    vocab: Dict[str, Dict[str, str]] = {}

    for sop in sops:
        for tag in sop.applies_to:
            if tag.lower() == "any":
                continue

            tag_key = tag.strip().lower()
            if tag_key not in vocab:
                # Initialize vocabulary entry
                title = sop.title
                description = sop.intent_description or f"Safety rules and guidance for {tag_key} activities."
                vocab[tag_key] = {
                    "title": title,
                    "intent_description": description,
                }
            else:
                # If a semantic SOP has a richer intent description, use or combine it
                if sop.intent_description and "guidance for" in vocab[tag_key]["intent_description"]:
                    vocab[tag_key]["intent_description"] = sop.intent_description

    return vocab
