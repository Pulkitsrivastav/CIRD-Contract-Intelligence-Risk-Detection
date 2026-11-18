"""Human-in-the-loop review helpers for clause findings."""
from __future__ import annotations

from copy import deepcopy

from analyzer import calculate_risk_score


def apply_human_reviews(analysis_result: dict, overrides: dict, standards_list: list[dict]) -> dict:
    """Return a reviewed copy without mutating the original model/rule findings."""
    reviewed = deepcopy(analysis_result)
    any_reviewed = False
    for clause in reviewed.get("clauses", []):
        override = overrides.get(clause.get("clause_id"))
        if not override:
            clause["human_reviewed"] = False
            continue
        status = override.get("status", clause["status"])
        risk = override.get("risk_level", clause["risk_level"])
        note = str(override.get("note", "")).strip()
        valid_status = {"Present", "Needs Review", "Missing / Not Detected"}
        valid_risk = {"Low Risk", "Medium Risk", "High Risk"}
        if status not in valid_status or risk not in valid_risk:
            continue
        if status == "Missing / Not Detected":
            risk = "High Risk"
        if status == "Present" and risk == "High Risk":
            status = "Needs Review"
        if status == "Needs Review" and risk == "Low Risk":
            risk = "Medium Risk"
        clause["pre_review_status"] = clause["status"]
        clause["pre_review_risk_level"] = clause["risk_level"]
        clause["status"] = status
        clause["risk_level"] = risk
        clause["engine_fuzzy_risk_score"] = clause.get("fuzzy_risk_score")
        clause["human_review_score"] = {"Low Risk": 10.0, "Medium Risk": 50.0, "High Risk": 90.0}[risk]
        clause["human_reviewed"] = True
        clause["human_review_note"] = note
        clause["csp_solved"] = True
        clause["csp_note"] = "Human-reviewed status/risk pair normalized to satisfy the same consistency constraints."
        any_reviewed = True
    if any_reviewed:
        reviewed["score_summary"] = calculate_risk_score(reviewed["clauses"], standards_list)
        reviewed["score_summary"]["formula_explanation"] += (
            " Human-reviewed clauses use the reviewer-selected Low/Medium/High category value (10/50/90) "
            "in place of the engine fuzzy output; unreviewed clauses retain their original fuzzy score."
        )
        reviewed["score_summary"]["score_basis"] = "Mixed: engine fuzzy inference plus human-reviewed category overrides"
        reviewed["human_review_applied"] = True
    else:
        reviewed["human_review_applied"] = False
    return reviewed
