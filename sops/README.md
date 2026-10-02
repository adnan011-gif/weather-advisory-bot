# Standard Operating Procedures (SOPs)

This directory contains declarative, human-auditable weather safety rules.
Each file specifies a single Standard Operating Procedure evaluated deterministically against computed Open-Meteo meteorological facts.

## SOP YAML Schema

```yaml
id: <UNIQUE_ID>                     # e.g., EXR-001
title: "<HUMAN_READABLE_TITLE>"      # e.g., High Heat Running Restriction
category: "<CATEGORY>"              # outdoor_exercise | travel | vulnerable_groups | leisure | general
severity: "<SEVERITY>"              # info | low | moderate | high | critical
priority: <INT>                     # Lower number ranks higher (1 = highest)
applies_to:                         # Activity/scenario tags or ["any"]
  - "<TAG>"
match_type: "<MATCH_TYPE>"          # numeric | composite | semantic | clear
override: false                     # true if this rule supersedes all non-override SOPs
condition:                          # Nested all_of / any_of or leaf condition {fact, op, value}
  fact: "<FACT_NAME>"               # Must exist in config/facts.yaml
  op: ">="                          # >, >=, <, <=, ==, !=, in, intersects
  value: <TARGET_VALUE>
advice: "<ADVICE_TEXT>"             # Text with optional {fact_name} placeholders
rationale: "<RATIONALE>"            # Explanation of the safety threshold
```

---

## How to Add an SOP in 5 Lines

To add a new rule without modifying any Python code, create a new YAML file in `sops/` (e.g. `sops/EXR-004.yaml`):

```yaml
id: EXR-004
title: "Cold Weather Running Advisory"
category: "outdoor_exercise"
severity: "moderate"
priority: 10
applies_to: ["running", "outdoor_exercise"]
match_type: "numeric"
condition: {fact: "temperature", op: "<=", value: 0.0}
advice: "Freezing conditions at {temperature}°C. Risk of hypothermia and icy paths; dress in thermal layers."
rationale: "Policy choice: Sub-zero temperatures present cold-injury risks."
```
Once saved, the system automatically detects the new SOP and incorporates its activity tags into the intent classification vocabulary.
