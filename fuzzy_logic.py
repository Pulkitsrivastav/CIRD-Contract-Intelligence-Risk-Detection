"""Small, dependency-free Mamdani fuzzy inference system for clause risk.

Inputs are lexical coverage (0..1) and red-flag strength (0..1).  The module
is an explainable academic screening aid, not a legally calibrated model.
"""
from __future__ import annotations

from typing import Dict, Iterable


def _trap(x: float, a: float, b: float, c: float, d: float) -> float:
    if x < a or x > d:
        return 0.0
    if a <= x <= b:
        return 1.0 if a == b else (x - a) / (b - a)
    if b < x <= c:
        return 1.0
    if c < x <= d:
        return 1.0 if c == d else (d - x) / (d - c)
    return 0.0


def _triangle(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 1.0 if x == b and a == b == c else 0.0
    if x == b:
        return 1.0
    return (x - a) / (b - a) if x < b and b != a else (c - x) / (c - b)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def infer_clause_risk(coverage_ratio: float, red_flag_strength: float) -> Dict[str, object]:
    """Return a 0..100 fuzzy risk score, memberships and an explanation.

    Mamdani rules:
      1. Low coverage -> High risk.
      2. High red-flag strength -> High risk.
      3. Medium coverage -> Medium risk.
      4. High coverage AND low red-flag strength -> Low risk.
      5. High coverage AND medium red-flag strength -> Medium risk.
    Rule firing uses min; output sets are aggregated using max; centroid
    defuzzification produces the crisp score.
    """
    coverage = _clip(coverage_ratio)
    flags = _clip(red_flag_strength)

    # Lexical coverage is intentionally calibrated generously because legal
    # drafting uses many valid synonyms that a small keyword list cannot capture.
    cov_low = _trap(coverage, 0.0, 0.0, 0.05, 0.20)
    cov_medium = _triangle(coverage, 0.10, 0.22, 0.34)
    cov_high = _trap(coverage, 0.25, 0.42, 1.0, 1.0)

    flag_low = _trap(flags, 0.0, 0.0, 0.10, 0.30)
    flag_medium = _triangle(flags, 0.15, 0.45, 0.75)
    flag_high = _trap(flags, 0.55, 0.75, 1.0, 1.0)

    high_strength = max(cov_low, flag_high)
    medium_strength = max(cov_medium, min(cov_high, flag_medium))
    low_strength = min(cov_high, flag_low)

    universe = list(range(0, 101))
    aggregated = []
    for x in universe:
        low_set = _trap(float(x), 0.0, 0.0, 20.0, 42.0)
        medium_set = _triangle(float(x), 25.0, 50.0, 75.0)
        high_set = _trap(float(x), 58.0, 78.0, 100.0, 100.0)
        mu = max(
            min(low_strength, low_set),
            min(medium_strength, medium_set),
            min(high_strength, high_set),
        )
        aggregated.append(mu)

    denominator = sum(aggregated)
    crisp = (sum(x * mu for x, mu in zip(universe, aggregated)) / denominator) if denominator else 50.0
    score = round(_clip(crisp, 0.0, 100.0), 1)
    if score <= 30:
        label = "Low Risk"
    elif score <= 64:
        label = "Medium Risk"
    else:
        label = "High Risk"

    return {
        "fuzzy_risk_score": score,
        "fuzzy_risk_level": label,
        "input_coverage": round(coverage, 3),
        "input_red_flag_strength": round(flags, 3),
        "membership_degrees": {
            "coverage": {"low": round(cov_low, 3), "medium": round(cov_medium, 3), "high": round(cov_high, 3)},
            "red_flags": {"low": round(flag_low, 3), "medium": round(flag_medium, 3), "high": round(flag_high, 3)},
            "risk_rules": {"low": round(low_strength, 3), "medium": round(medium_strength, 3), "high": round(high_strength, 3)},
        },
        "method": "Mamdani min-max inference with centroid defuzzification",
    }


def weighted_fuzzy_score(clause_results: Iterable[dict], standards: Iterable[dict]) -> float:
    weights = {item["id"]: float(item.get("weight", 1.0)) for item in standards}
    pairs = [
        (
            float(c.get("human_review_score", c.get("fuzzy_risk_score", 50.0)))
            if c.get("human_reviewed")
            else float(c.get("fuzzy_risk_score", 50.0)),
            weights.get(c.get("clause_id"), 1.0),
        )
        for c in clause_results
    ]
    total_weight = sum(weight for _, weight in pairs)
    return round(sum(score * weight for score, weight in pairs) / total_weight, 1) if total_weight else 0.0
