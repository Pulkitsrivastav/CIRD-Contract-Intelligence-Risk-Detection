"""Built-in deterministic regression checks for the supplied sample NDAs.

These are functionality checks against intentionally authored examples, not a
statistically representative accuracy benchmark or legal validation dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from analyzer import analyze_with_keywords, load_standard_clauses
from document_utils import extract_text_from_pdf


def run_sample_evaluation(project_root: str = ".") -> dict:
    root = Path(project_root)
    standards = load_standard_clauses(str(root / "data" / "nda_standards.json"))
    cases = [
        {
            "name": "Balanced standard mutual NDA",
            "filename": "standard_mutual_nda.pdf",
            "checks": [
                ("all ten clause topics are labelled Present", lambda r: all(c["status"] == "Present" for c in r["clauses"])),
                ("overall risk remains Low", lambda r: r["score_summary"]["overall_category"] == "Low Risk"),
                ("no configured red flags are detected", lambda r: all(not c.get("matched_red_flags") for c in r["clauses"])),
            ],
        },
        {
            "name": "Intentionally flawed mutual NDA",
            "filename": "high_risk_flawed_nda.pdf",
            "checks": [
                ("confidentiality period D is flagged", lambda r: _clause(r, "D")["risk_level"] == "High Risk"),
                ("liability and remedies H is flagged", lambda r: _clause(r, "H")["risk_level"] == "High Risk"),
                ("missing exclusions, return/destruction, disclosure and security clauses are detected", lambda r: all(_clause(r, cid)["status"] == "Missing / Not Detected" for cid in ("C", "E", "F", "G"))),
                ("overall risk is High", lambda r: r["score_summary"]["overall_category"] == "High Risk"),
            ],
        },
    ]

    output_cases = []
    all_checks = []
    for case in cases:
        path = root / "sample_documents" / case["filename"]
        if not path.exists():
            output_cases.append({"name": case["name"], "success": False, "error": f"Missing sample file: {path}", "checks": []})
            continue
        extracted = extract_text_from_pdf(path.read_bytes(), path.name)
        if not extracted.get("success"):
            output_cases.append({"name": case["name"], "success": False, "error": extracted.get("error"), "checks": []})
            continue
        result = analyze_with_keywords(extracted["full_text"], standards)
        case_checks = []
        for label, check in case["checks"]:
            passed = bool(check(result))
            case_checks.append({"check": label, "passed": passed})
            all_checks.append(passed)
        output_cases.append({
            "name": case["name"],
            "filename": case["filename"],
            "success": all(item["passed"] for item in case_checks),
            "overall_score": result["score_summary"]["overall_score"],
            "overall_category": result["score_summary"]["overall_category"],
            "clauses_reviewed": result["score_summary"]["clauses_reviewed"],
            "checks": case_checks,
        })

    passed = sum(all_checks)
    total = len(all_checks)
    return {
        "cases": output_cases,
        "checks_passed": passed,
        "checks_total": total,
        "check_pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "interpretation": "Pass rate is the share of predefined regression checks passed on the two bundled sample PDFs. It is not model accuracy, generalization performance, or legal validation.",
    }


def _clause(result: dict, clause_id: str) -> dict:
    return next(clause for clause in result["clauses"] if clause["clause_id"] == clause_id)
