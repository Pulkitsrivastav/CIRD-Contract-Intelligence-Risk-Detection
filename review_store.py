"""Persistent human-approval workflow for organization-specific NDA examples.

Storage is a small JSON document so the academic project works without a database.
The permanent baseline and semi-approved queue are deliberately separate.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

DEFAULT_STORE_PATH = Path(__file__).resolve().parent / "data" / "human_approval_store.json"
RISK_LEVELS = {"Low Risk", "Medium Risk", "High Risk"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty_store() -> dict:
    return {
        "schema_version": 1,
        "approved_baseline": [],
        "pending_approvals": [],
        "dismissed_items": [],
        "audit_log": [],
    }


def load_review_store(path: str | os.PathLike = DEFAULT_STORE_PATH) -> dict:
    """Load the persistent store, repairing missing keys but not hiding corrupt JSON."""
    target = Path(path)
    if not target.exists():
        return _empty_store()
    with target.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Human approval store must contain a JSON object.")
    base = _empty_store()
    base.update(data)
    for key in ("approved_baseline", "pending_approvals", "dismissed_items", "audit_log"):
        if not isinstance(base.get(key), list):
            base[key] = []
    return base


def save_review_store(store: dict, path: str | os.PathLike = DEFAULT_STORE_PATH) -> None:
    """Atomically write review data to disk so partial writes do not corrupt it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(store, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def normalize_text(value: str) -> str:
    value = re.sub(r"^\s*Page\s+\d+\s*:\s*", "", str(value or ""), flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", value.lower())).strip()


def _similarity(left: str, right: str) -> float:
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _candidate_id(document_hash: str, source_type: str, identity: str, text: str) -> str:
    raw = "|".join((document_hash, source_type, identity, normalize_text(text)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _clean_evidence(clause: dict) -> str:
    evidence = str(clause.get("extracted_evidence") or "").strip()
    if evidence.lower().startswith("no matching keywords") or evidence.lower() in {"not found in contract", ""}:
        return ""
    return re.sub(r"^Page\s+\d+\s*:\s*", "", evidence, flags=re.I).strip()


def _extract_additional_sections(contract_text: str, standards: list[dict]) -> list[dict]:
    """Find numbered headings outside the ten fixed baseline categories.

    All numbered headings are used as section boundaries, including standard
    headings that are not themselves sent to human approval. This prevents an
    additional provision's evidence from accidentally swallowing later clauses.
    """
    known_terms = {
        "definition of confidential information", "permitted use and purpose", "permitted use",
        "exclusions from confidentiality", "confidentiality period", "confidentiality duration",
        "return or destruction of information", "return or destruction", "permitted disclosure",
        "permitted disclosure and legal compulsion", "security and safeguarding", "security",
        "liability and remedies", "remedies and liability", "governing law and jurisdiction",
        "governing law", "termination",
    }
    for item in standards:
        known_terms.add(normalize_text(str(item.get("name", ""))))

    page = 1
    lines = str(contract_text or "").splitlines()
    all_headings: list[tuple[int, str, int, bool]] = []
    for index, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("--- Page ") and line.endswith(" ---"):
            match = re.search(r"Page\s+(\d+)", line)
            if match:
                page = int(match.group(1))
            continue
        match = re.match(r"^\s*(\d+(?:\.\d+)*[.)]?)\s+([A-Z][A-Za-z0-9 ,/&()'’-]{2,85})\s*$", line)
        if match:
            title = match.group(2).strip(" .:-")
        else:
            article_match = re.match(
                r"^\s*(?:ARTICLE|SECTION)\s+[A-Z0-9IVX.-]+\s*(?:[:.)\-–—]+)?\s+(.{3,85})$",
                line,
                re.IGNORECASE,
            )
            if not article_match:
                continue
            title = article_match.group(1).strip(" .:-")
        normalized = normalize_text(title)
        if not normalized:
            continue
        is_additional = normalized not in known_terms
        if any(term in normalized for term in (
            "non disclosure agreement", "mutual nda", "sample document",
            "confidential information", "permitted use", "confidentiality period",
            "confidentiality duration", "exclusions", "return or destruction",
            "permitted disclosure", "security and safeguarding", "liability",
            "remedies", "governing law", "jurisdiction", "termination",
        )):
            is_additional = False
        all_headings.append((index, title, page, is_additional))

    candidates = []
    for position, (line_index, title, page_number, is_additional) in enumerate(all_headings):
        if not is_additional:
            continue
        next_index = all_headings[position + 1][0] if position + 1 < len(all_headings) else len(lines)
        body_lines = []
        for line in lines[line_index + 1:next_index]:
            if line.strip().startswith("--- Page "):
                continue
            if line.strip():
                body_lines.append(line.strip())
        body = " ".join(body_lines).strip()
        if len(body) < 45:
            continue
        evidence = f"{title}: {body}"
        if len(evidence) > 1400:
            evidence = evidence[:1397].rstrip() + "..."
        candidates.append({"title": title, "page_number": page_number, "evidence": evidence})
    return candidates


def build_review_candidates(
    analysis_result: dict,
    contract_text: str,
    document_hash: str,
    filename: str,
    store: dict,
    standards: list[dict],
) -> dict:
    """Separate auto-matched findings from items that require human approval."""
    approved = store.get("approved_baseline", [])
    pending_records = store.get("pending_approvals", [])
    pending_ids = {item.get("id") for item in pending_records}
    dismissed_ids = {item.get("id") for item in store.get("dismissed_items", [])}
    candidates: list[dict] = []
    auto_approved: list[dict] = []
    recognized_additional: list[dict] = []

    for clause in analysis_result.get("clauses", []):
        evidence = _clean_evidence(clause)
        matched = None
        if evidence:
            for item in approved:
                if item.get("source_type") == "baseline_clause_deviation" and item.get("clause_id") == clause.get("clause_id") and _similarity(evidence, item.get("pattern_text", "")) >= 0.88:
                    matched = item
                    break
        if matched:
            auto_approved.append({
                "title": clause.get("clause_name"),
                "clause_id": clause.get("clause_id"),
                "reason": f"Matches a previously human-approved baseline example ({matched.get('approved_risk_level', 'Medium Risk')}).",
            })
            continue
        needs_review = clause.get("status") != "Present" or clause.get("risk_level") != "Low Risk"
        if not needs_review:
            auto_approved.append({"title": clause.get("clause_name"), "clause_id": clause.get("clause_id"), "reason": "Detected and within the current reference baseline."})
            continue

        candidate_id = _candidate_id(document_hash, "baseline_clause_deviation", clause.get("clause_id", ""), evidence or clause.get("clause_name", ""))
        if candidate_id in pending_ids or candidate_id in dismissed_ids:
            continue
        already_queued = any(
            queued.get("source_type") == "baseline_clause_deviation"
            and queued.get("clause_id") == clause.get("clause_id")
            and evidence and _similarity(evidence, queued.get("evidence", "")) >= 0.88
            for queued in pending_records
        )
        if already_queued:
            continue
        candidates.append({
            "id": candidate_id,
            "source_type": "baseline_clause_deviation",
            "title": clause.get("clause_name", "Unnamed clause"),
            "clause_id": clause.get("clause_id"),
            "page_number": clause.get("evidence_page"),
            "status": clause.get("status"),
            "risk_level": clause.get("risk_level"),
            "evidence": evidence,
            "explanation": clause.get("difference_explanation", ""),
            "suggested_action": clause.get("suggested_action", ""),
            "filename": filename,
            "document_hash": document_hash,
            "created_at": _now(),
            "can_add_to_baseline": bool(evidence),
        })

    for section in _extract_additional_sections(contract_text, standards):
        approved_match = next((item for item in approved if item.get("source_type") == "additional_clause" and _similarity(section["evidence"], item.get("pattern_text", "")) >= 0.88), None)
        if approved_match:
            recognized_additional.append({
                "title": section["title"],
                "page_number": section["page_number"],
                "reason": "Matches a previously human-approved additional clause pattern.",
            })
            continue
        candidate_id = _candidate_id(document_hash, "additional_clause", section["title"], section["evidence"])
        if candidate_id in pending_ids or candidate_id in dismissed_ids:
            continue
        already_queued = any(
            queued.get("source_type") == "additional_clause"
            and _similarity(section["evidence"], queued.get("evidence", "")) >= 0.88
            for queued in pending_records
        )
        if already_queued:
            continue
        candidates.append({
            "id": candidate_id,
            "source_type": "additional_clause",
            "title": section["title"],
            "clause_id": None,
            "page_number": section["page_number"],
            "status": "New / Not in Baseline",
            "risk_level": "Needs Human Assessment",
            "evidence": section["evidence"],
            "explanation": "This numbered contract section does not map directly to one of the ten current baseline categories.",
            "suggested_action": "Decide whether this provision is useful and should become an organization-approved example.",
            "filename": filename,
            "document_hash": document_hash,
            "created_at": _now(),
            "can_add_to_baseline": True,
        })

    return {
        "auto_approved": auto_approved,
        "candidates": candidates,
        "recognized_additional": recognized_additional,
    }


def _record_pattern(item: dict, approved_risk_level: str, source: str) -> dict:
    if approved_risk_level not in {"Low Risk", "Medium Risk"}:
        raise ValueError("A permanent baseline approval must be rated Low Risk or Medium Risk.")
    if not str(item.get("evidence", "")).strip():
        raise ValueError("This item has no extracted clause text and cannot be added to the permanent baseline.")
    return {
        "id": item["id"],
        "source_type": item.get("source_type"),
        "clause_id": item.get("clause_id"),
        "title": item.get("title", "Unnamed clause"),
        "pattern_text": item["evidence"],
        "approved_risk_level": approved_risk_level,
        "source_filename": item.get("filename", "Unknown document"),
        "reviewer_note": item.get("reviewer_note", ""),
        "approved_at": _now(),
        "approval_source": source,
    }


def approve_permanently(item: dict, approved_risk_level: str, store: dict, path=DEFAULT_STORE_PATH) -> dict:
    pattern = _record_pattern(item, approved_risk_level, "direct_human_approval")
    store["approved_baseline"] = [x for x in store.get("approved_baseline", []) if x.get("id") != pattern["id"]]
    store["approved_baseline"].append(pattern)
    store["pending_approvals"] = [x for x in store.get("pending_approvals", []) if x.get("id") != pattern["id"]]
    store.setdefault("audit_log", []).append({"action": "permanently_approved", "item_id": pattern["id"], "title": pattern["title"], "at": _now()})
    save_review_store(store, path)
    return pattern


def semi_approve(item: dict, store: dict, path=DEFAULT_STORE_PATH) -> dict:
    queued = dict(item)
    queued["queue_status"] = "Semi-approved — awaiting baseline decision"
    queued["queued_at"] = _now()
    existing = {x.get("id") for x in store.get("pending_approvals", [])}
    if queued["id"] not in existing:
        store.setdefault("pending_approvals", []).append(queued)
        store.setdefault("audit_log", []).append({"action": "semi_approved_to_queue", "item_id": queued["id"], "title": queued.get("title"), "at": _now()})
    save_review_store(store, path)
    return queued


def promote_pending_to_baseline(item_id: str, approved_risk_level: str, store: dict, path=DEFAULT_STORE_PATH) -> dict:
    item = next((x for x in store.get("pending_approvals", []) if x.get("id") == item_id), None)
    if item is None:
        raise KeyError("The selected item is no longer in the approval queue.")
    pattern = _record_pattern(item, approved_risk_level, "promoted_from_semi_approval")
    store["approved_baseline"] = [x for x in store.get("approved_baseline", []) if x.get("id") != item_id]
    store["approved_baseline"].append(pattern)
    store["pending_approvals"] = [x for x in store.get("pending_approvals", []) if x.get("id") != item_id]
    store.setdefault("audit_log", []).append({"action": "promoted_to_baseline", "item_id": item_id, "title": item.get("title"), "at": _now()})
    save_review_store(store, path)
    return pattern



def reject_remove(item: dict, store: dict, path=DEFAULT_STORE_PATH) -> bool:
    """Dismiss this finding without changing the permanent baseline or pending queue."""
    dismissed = store.setdefault("dismissed_items", [])
    if any(existing.get("id") == item.get("id") for existing in dismissed):
        return False
    dismissed.append({
        "id": item.get("id"),
        "title": item.get("title"),
        "document_hash": item.get("document_hash"),
        "dismissed_at": _now(),
    })
    store.setdefault("audit_log", []).append({"action": "rejected_or_removed", "item_id": item.get("id"), "title": item.get("title"), "at": _now()})
    save_review_store(store, path)
    return True


def remove_pending(item_id: str, store: dict, path=DEFAULT_STORE_PATH) -> bool:
    before = len(store.get("pending_approvals", []))
    store["pending_approvals"] = [x for x in store.get("pending_approvals", []) if x.get("id") != item_id]
    removed = len(store["pending_approvals"]) != before
    if removed:
        store.setdefault("audit_log", []).append({"action": "removed_from_queue", "item_id": item_id, "at": _now()})
        save_review_store(store, path)
    return removed


def match_approved_clause_pattern(clause: dict, store: dict) -> dict | None:
    """Return an approved baseline pattern matching this clause's source evidence."""
    evidence = _clean_evidence(clause)
    if not evidence:
        return None
    for item in store.get("approved_baseline", []):
        if item.get("source_type") != "baseline_clause_deviation":
            continue
        if item.get("clause_id") != clause.get("clause_id"):
            continue
        if _similarity(evidence, item.get("pattern_text", "")) >= 0.88:
            return item
    return None


def apply_approved_baseline_matches(analysis_result: dict, store: dict, standards: list[dict]) -> dict:
    """Apply prior permanent human approvals to materially matching future clauses."""
    from copy import deepcopy
    from analyzer import calculate_risk_score

    reviewed = deepcopy(analysis_result)
    score_points = {"Low Risk": 10.0, "Medium Risk": 50.0}
    changed = False
    for clause in reviewed.get("clauses", []):
        match = match_approved_clause_pattern(clause, store)
        if not match or clause.get("status") == "Missing / Not Detected":
            continue
        accepted_level = match.get("approved_risk_level", "Medium Risk")
        if accepted_level not in score_points:
            continue
        clause["pre_baseline_match_status"] = clause.get("status")
        clause["pre_baseline_match_risk"] = clause.get("risk_level")
        clause["status"] = "Present"
        clause["risk_level"] = accepted_level
        clause["engine_fuzzy_risk_score"] = clause.get("fuzzy_risk_score")
        clause["fuzzy_risk_score"] = score_points[accepted_level]
        clause["fuzzy_risk_level"] = accepted_level
        clause["csp_note"] = "Matched a permanently human-approved example; the accepted risk category is applied for this future match."
        clause["baseline_approved_match"] = {
            "baseline_item_id": match.get("id"),
            "approved_title": match.get("title"),
            "approved_risk_level": accepted_level,
            "approved_at": match.get("approved_at"),
        }
        clause["baseline_match_note"] = "This wording materially matches a pattern permanently approved by a human reviewer. The stored accepted risk rating is applied for future matching."
        changed = True
    if changed:
        reviewed["score_summary"] = calculate_risk_score(reviewed.get("clauses", []), standards)
        reviewed["executive_summary"] = str(reviewed.get("executive_summary", "")) + " Previously human-approved baseline examples were applied where materially similar wording was detected."
        reviewed["baseline_approvals_applied"] = True
    else:
        reviewed["baseline_approvals_applied"] = False
    return reviewed
