"""
analyzer.py
Core NDA Clause Identification, Reference Comparison, Risk Scoring, and Report Generation.
All contract analysis is performed locally by the offline contextual screening engine.
"""

import json
import os
import re
from datetime import datetime

from fuzzy_logic import infer_clause_risk
from csp_solver import reconcile_clause_labels, validate_clause_assignment
from state_space import plan_analysis_route
from learned_clause_model import model_status

# Transparent Risk Scoring Constants
RISK_POINTS = {
    "Low Risk": 10,
    "Medium Risk": 50,
    "High Risk": 90,
}


def load_standard_clauses(standards_path: str = "data/nda_standards.json") -> dict:
    """Loads the predefined standard NDA reference library from JSON."""
    if not os.path.exists(standards_path):
        raise FileNotFoundError(f"Standard NDA reference library not found at: {standards_path}")
    with open(standards_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_risk_score(clause_results: list[dict], standards_list: list[dict]) -> dict:
    """Calculate the primary score from final clause risk labels.

    The fuzzy score remains visible as a diagnostic, but it cannot override a
    High/Medium/Low clause label in the headline dashboard score.
    """
    weight_map = {item["id"]: float(item.get("weight", 1.0)) for item in standards_list}
    weighted_sum = 0.0
    total_weight = 0.0
    fuzzy_weighted_sum = 0.0
    high_count = medium_count = low_count = 0
    present_count = needs_review_count = missing_count = 0

    for clause in clause_results:
        cid = clause.get("clause_id", "")
        weight = weight_map.get(cid, 1.0)
        risk = clause.get("risk_level", "Medium Risk")
        status = clause.get("status", "Needs Review")
        if risk == "High Risk":
            high_count += 1
        elif risk == "Medium Risk":
            medium_count += 1
        else:
            low_count += 1
        if status == "Present":
            present_count += 1
        elif status == "Missing / Not Detected":
            missing_count += 1
        else:
            needs_review_count += 1

        # Final risk labels use the declared 10/50/90 point scale.
        weighted_sum += RISK_POINTS.get(risk, 50) * weight
        total_weight += weight
        fuzzy_weighted_sum += float(clause.get("fuzzy_risk_score", 50.0)) * weight

    overall_score = round(weighted_sum / total_weight) if total_weight else 0
    fuzzy_score = round(fuzzy_weighted_sum / total_weight, 1) if total_weight else 0.0
    # 60+ is High so a finding set dominated by high-risk clauses is not hidden
    # by a separate lexical-coverage diagnostic.
    if overall_score <= 30:
        overall_category = "Low Risk"
    elif overall_score <= 59:
        overall_category = "Medium Risk"
    else:
        overall_category = "High Risk"

    formula_note = (
        "Primary score: importance-weighted mean of final clause risk labels on a 10/50/90 point scale "
        "(Low/Medium/High). Mamdani fuzzy inference remains a separate diagnostic based on concept coverage "
        "and red-flag strength; it does not override final clause risk labels. Category thresholds: "
        "0–30 Low, 31–59 Medium, 60–100 High. This is an educational screening indicator, not a "
        "probability or legal opinion."
    )
    return {
        "overall_score": overall_score,
        "overall_category": overall_category,
        "fuzzy_overall_score": fuzzy_score,
        "legacy_weighted_score": overall_score,
        "clauses_reviewed": len(clause_results),
        "high_risk_count": high_count,
        "medium_risk_count": medium_count,
        "low_risk_count": low_count,
        "present_count": present_count,
        "needs_review_count": needs_review_count,
        "missing_count": missing_count,
        "formula_explanation": formula_note,
        "score_basis": "Importance-weighted final clause risk labels (10/50/90); fuzzy score shown separately",
    }


def _split_into_evidence_blocks(text: str) -> list[dict]:
    """Split text into page-aware, section-aware evidence blocks.

    PyMuPDF frequently returns line-wrapped text without blank lines. Section
    numbering is therefore used as an additional boundary, while wrapped
    continuation lines remain attached to their section.
    """
    page_pattern = re.compile(r"---\s*Page\s+(\d+)\s*---", re.IGNORECASE)
    matches = list(page_pattern.finditer(text or ""))
    pages = []
    if matches:
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            pages.append((int(match.group(1)), text[match.end():end]))
    else:
        pages = [(1, text or "")]

    heading_pattern = re.compile(r"^\s*(?:(?:\d+(?:\.\d+)*[.)]?)\s+|(?:ARTICLE|SECTION)\s+[A-Z0-9IVX]+\b)", re.IGNORECASE)
    blocks = []
    for page_number, page_text in pages:
        lines = [line.strip() for line in page_text.replace("\r", "\n").split("\n")]
        current = []
        for line in lines:
            if not line:
                if current:
                    joined = " ".join(current).strip()
                    if len(joined) >= 20:
                        blocks.append({"page_number": page_number, "text": joined})
                    current = []
                continue
            if heading_pattern.match(line) and current:
                joined = " ".join(current).strip()
                if len(joined) >= 20:
                    blocks.append({"page_number": page_number, "text": joined})
                current = []
            current.append(line)
        if current:
            joined = " ".join(current).strip()
            if len(joined) >= 20:
                blocks.append({"page_number": page_number, "text": joined})

    # If there are no useful structural boundaries, split long prose into
    # sentence groups so that evidence remains local rather than page-wide.
    if len(blocks) <= 1 and blocks and len(blocks[0]["text"]) > 900:
        source = blocks[0]
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", source["text"])
        rebuilt, buffer = [], ""
        for sentence in sentences:
            if buffer and len(buffer) + len(sentence) > 500:
                rebuilt.append({"page_number": source["page_number"], "text": buffer.strip()})
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            rebuilt.append({"page_number": source["page_number"], "text": buffer.strip()})
        blocks = rebuilt
    return blocks


def _split_into_paragraphs(text: str) -> list[str]:
    """Backward-compatible paragraph helper used by tests and integrations."""
    return [block["text"] for block in _split_into_evidence_blocks(text)]


def _red_flag_strength(matched_red_flags: list[str]) -> float:
    if not matched_red_flags:
        return 0.0
    severe_terms = ("perpetual", "indefinitely", "forever", "unlimited liability", "liquidated damages", "sole discretion")
    if any(any(term in flag for term in severe_terms) for flag in matched_red_flags):
        return 1.0
    return min(0.85, 0.45 + 0.15 * (len(matched_red_flags) - 1))


def _verify_evidence(evidence: str, contract_text: str) -> dict:
    """Check whether an asserted excerpt is traceable to extracted source text."""
    if not evidence or evidence.lower().startswith(("no matching", "not found", "no clause")):
        return {"verified": None, "verification_note": "No positive source excerpt was asserted."}
    normalize = lambda value: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value.lower())).strip()
    excerpt = normalize(evidence)
    source = normalize(contract_text)
    if not excerpt:
        return {"verified": False, "verification_note": "Evidence text was empty after normalization."}
    if excerpt in source:
        return {"verified": True, "verification_note": "Evidence excerpt matches extracted contract text."}
    # Compare short contiguous source spans conservatively; never treat a paraphrase as an exact quote.
    tokens = excerpt.split()
    if len(tokens) >= 8:
        windows = [" ".join(tokens[i:i + 8]) for i in range(len(tokens) - 7)]
        if any(window in source for window in windows):
            return {"verified": True, "verification_note": "A contiguous 8-word evidence span matches the source text."}
    return {"verified": False, "verification_note": "Could not verify this wording as an exact source excerpt; inspect the original PDF."}


def analyze_with_keywords(contract_text: str, standards_data: dict) -> dict:
    """Run local contextual rules, the bundled learned classifier, fuzzy logic and CSP.

    No API key, network access, cloud model or external inference service is used.
    """
    from offline_engine import analyze_offline

    standards_list = standards_data.get("clauses", [])
    blocks = _split_into_evidence_blocks(contract_text)
    if not blocks:
        return {"success": False, "error": "No usable text blocks were available for analysis."}

    offline = analyze_offline(contract_text, standards_data, blocks)
    clause_results = offline.get("clauses", [])
    score_summary = calculate_risk_score(clause_results, standards_list)

    priorities = []
    for clause in sorted(
        clause_results,
        key=lambda item: (
            item.get("risk_level") == "High Risk",
            item.get("risk_level") == "Medium Risk",
            item.get("raw_concept_coverage", 0),
        ),
        reverse=True,
    ):
        if clause.get("status") != "Present" or clause.get("risk_level") != "Low Risk":
            priorities.append(
                f"[{clause['status']} / {clause['risk_level']}] {clause['clause_name']} — {clause['suggested_action']}"
            )
        if len(priorities) >= 5:
            break
    if not priorities:
        priorities = [
            "All configured clause concepts were detected; a human reviewer should still verify the complete provisions and their legal effect."
        ]

    present = score_summary["present_count"]
    review = score_summary["needs_review_count"]
    missing = score_summary["missing_count"]
    return {
        "success": True,
        "error": None,
        "analysis_mode": "Offline-Only Contextual NLP + Locally Trained Clause Classifier + Fuzzy Risk + CSP",
        "executive_summary": (
            f"Offline screening assessed {len(clause_results)} NDA clause categories. "
            f"{present} were labelled Present, {review} Need Review and {missing} Not Detected. "
            "The local engine combines clause-specific contextual rules, a locally trained TF-IDF word/character n-gram classifier, "
            "negation-aware red-flag patterns, fuzzy diagnostics and CSP consistency checks. ML-only matches are routed to human review. "
            "The bundled training corpus is synthetic and educational, not a representative legal dataset; this is not a legal opinion."
        ),
        "top_review_priorities": priorities,
        "score_summary": score_summary,
        "clauses": clause_results,
        "method_details": {
            "offline_engine": offline.get("engine_name", "Context-aware Offline Screening Engine"),
            "offline_engine_version": offline.get("engine_version", "2.0"),
            "offline_method": "Clause-specific concept groups; phrase and guarded single-term patterns; context windows; a locally trained TF-IDF word/character n-gram classifier; section headings; red-flag combinations; negation checks; exact evidence excerpts.",
            "fuzzy_logic": "Mamdani min-max inference with centroid defuzzification using concept coverage and red-flag severity.",
            "csp": "Finite-domain status/risk consistency validation using AC-3 and backtracking.",
            "learned_classifier": model_status(),
            "contextual_matching": True,
            "multi_sentence_matching": True,
            "negation_aware_flags": True,
            "state_space_path": plan_analysis_route(),
        },
    }


def analyze_contract(contract_text: str, standards_data: dict) -> dict:
    """Analyze a contract using the offline engine only. No network/API path exists."""
    return analyze_with_keywords(contract_text, standards_data)


def generate_markdown_report(analysis_result: dict, doc_info: dict) -> str:
    """Generates a clean, structured Markdown contract analysis report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score = analysis_result["score_summary"]
    clauses = analysis_result["clauses"]

    lines = [
        "# AI Contract Intelligence & Risk Detection System — Analysis Report",
        "",
        f"- **Generated On:** {timestamp}",
        f"- **Contract File:** `{doc_info.get('filename', 'Unknown.pdf')}`",
        f"- **Document Stats:** {doc_info.get('page_count', 0)} page(s) | {doc_info.get('word_count', 0)} words",
        f"- **Analysis Engine:** {analysis_result.get('analysis_mode', 'Standard Screening')}",
        "",
        "---",
        "",
        "## 1. Overall Risk Assessment",
        "",
        f"- **Overall Risk Score:** **{score['overall_score']} / 100** ({score['overall_category']})",
        f"- **Clauses Reviewed:** {score['clauses_reviewed']}",
        f"- **High-Risk Findings:** {score['high_risk_count']}",
        f"- **Medium-Risk Findings:** {score['medium_risk_count']}",
        f"- **Low-Risk Findings:** {score['low_risk_count']}",
        f"- **Status Breakdown:** {score['present_count']} Present | {score['needs_review_count']} Needs Review | {score['missing_count']} Missing / Not Detected",
        "",
        f"> **Scoring Methodology:** {score['formula_explanation']}",
        f"> **Score basis:** {score.get('score_basis', 'Weighted Mamdani fuzzy screening score')}",
        f"> **Fuzzy diagnostic (separate):** {score.get('fuzzy_overall_score', 'N/A')}/100",
        "",
        "## 2. Executive Summary",
        "",
        analysis_result.get("executive_summary", ""),
        "",
        "## 3. Recommended Review Priorities",
        "",
    ]

    for idx, priority in enumerate(analysis_result.get("top_review_priorities", []), start=1):
        lines.append(f"{idx}. {priority}")

    approval_summary = analysis_result.get("approval_workflow_summary", {})
    if approval_summary:
        lines.extend([
            "",
            "## 4. Human Approval Workflow Summary",
            "",
            f"- **Baseline-matched / auto-approved:** {approval_summary.get('auto_approved_count', 0)}",
            f"- **Awaiting human approval:** {approval_summary.get('awaiting_human_approval_count', 0)}",
            f"- **Semi-approved queue:** {approval_summary.get('semi_approved_queue_count', 0)}",
            f"- **Permanent supplemental baseline examples:** {approval_summary.get('permanent_supplemental_baseline_count', 0)}",
            "- Permanent approvals are stored as supplemental human-approved examples; the original core educational baseline is not overwritten.",
        ])
        clause_section_number = 5
    else:
        clause_section_number = 4

    lines.extend([
        "",
        f"## {clause_section_number}. Clause-by-Clause Summary Matrix",
        "",
        "| ID | Clause Category | Status | Risk Level |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for c in clauses:
        lines.append(f"| {c['clause_id']} | {c['clause_name']} | {c['status']} | {c['risk_level']} |")

    lines.extend([
        "",
        f"## {clause_section_number + 1}. Detailed Clause Comparison & Risk Explanations",
        "",
    ])

    for c in clauses:
        lines.extend([
            f"### {c['clause_id']}. {c['clause_name']} — [{c['status']} | {c['risk_level']}]",
            "",
            f"- **Extracted Contract Wording / Evidence:** {c['extracted_evidence']}",
            f"- **Evidence Verification:** {c.get('verification_note', 'Not assessed')}",
            *([f"- **Engine Fuzzy Risk Score (before review):** {c.get('engine_fuzzy_risk_score', c.get('fuzzy_risk_score', 'N/A'))} / 100 ({c.get('fuzzy_risk_level', 'N/A')})",
               f"- **Human-Reviewed Category Override:** {c.get('risk_level')} ({c.get('human_review_score', 'N/A')} / 100)"] if c.get('human_reviewed') else [f"- **Fuzzy Risk Score:** {c.get('fuzzy_risk_score', 'N/A')} / 100 ({c.get('fuzzy_risk_level', 'N/A')})"]),
            f"- **Standard Baseline Reference:** {c['standard_baseline']}",
            f"- **Comparison & Difference Explanation:** {c['difference_explanation']}",
            f"- **Potential Implication:** {c['potential_implication']}",
            f"- **Suggested Action for Review:** {c['suggested_action']}",
            *([f"- **Human Review:** {c.get('pre_review_status')} / {c.get('pre_review_risk_level')} → {c['status']} / {c['risk_level']}",
               f"- **Reviewer Note:** {c.get('human_review_note', '')}"] if c.get('human_reviewed') else []),
            "",
        ])

    missing_list = [c["clause_name"] for c in clauses if c["status"] == "Missing / Not Detected"]
    lines.extend([
        f"## {clause_section_number + 2}. Identified Missing Clauses",
        "",
    ])
    if missing_list:
        for m in missing_list:
            lines.append(f"- {m}")
    else:
        lines.append("- None. All 10 standard NDA clause categories were detected.")

    lines.extend([
        "",
        "---",
        "",
        "## Important Limitations & Legal Disclaimer",
        "",
        "- This report was generated by an academic preliminary contract screening tool (**AI Contract Intelligence & Risk Detection System**).",
        "- Risk scores and classifications are screening indicators based on an educational NDA baseline and do **not** represent a legally validated probability of legal risk.",
        "- This tool does **not** constitute legal advice and does **not** replace review by a qualified attorney.",
        "- Contract wording and jurisdiction-specific drafting may not fit this simplified educational NDA baseline; false positives and missed clauses are possible.",
        "- Contract text is processed locally by the offline screening engine; the application does not send contract content to an external AI API.",
        "- The local pattern engine can miss unusual wording or produce false positives. Verify evidence against the original PDF and retain human oversight.",
    ])

    return "\n".join(lines)


def generate_json_report(analysis_result: dict, doc_info: dict) -> str:
    """Generates a structured JSON analysis report for programmatic inspection or export."""
    payload = {
        "application": "AI Contract Intelligence & Risk Detection System",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "document_metadata": {
            "filename": doc_info.get("filename"),
            "page_count": doc_info.get("page_count"),
            "empty_pages": doc_info.get("empty_pages"),
            "word_count": doc_info.get("word_count"),
            "char_count": doc_info.get("char_count"),
        },
        "analysis_mode": analysis_result.get("analysis_mode"),
        "overall_assessment": analysis_result.get("score_summary"),
        "executive_summary": analysis_result.get("executive_summary"),
        "recommended_review_priorities": analysis_result.get("top_review_priorities"),
        "method_details": analysis_result.get("method_details", {}),
        "approval_workflow_summary": analysis_result.get("approval_workflow_summary", {}),
        "missing_clauses": [
            c["clause_name"]
            for c in analysis_result.get("clauses", [])
            if c["status"] == "Missing / Not Detected"
        ],
        "clause_comparisons": analysis_result.get("clauses", []),
        "disclaimer": (
            "Preliminary academic screening output only. Not authoritative legal advice. "
            "Consult a qualified legal professional before executing binding contracts."
        ),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
