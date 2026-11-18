"""Hybrid local NDA screening engine.

Combines transparent contextual rules with a locally trained TF-IDF classifier.
The learned model is a cautious second opinion; ML-only matches are sent to
human review and are never automatically approved.
"""
from __future__ import annotations

import re
from typing import Any

from csp_solver import validate_clause_assignment
from fuzzy_logic import infer_clause_risk
from learned_clause_model import predict_clause

# Each concept group represents one distinct obligation/idea in a clause.
# Patterns intentionally avoid treating generic words (e.g. "security", "years",
# "breach") as sufficient evidence on their own.
RULES: dict[str, dict[str, Any]] = {
    "A": {
        "heading": r"definition|confidential information|scope of confidential",
        "groups": {
            "confidential_scope": [r"confidential information", r"confidential material", r"protected information", r"non[- ]public (?:information|data)", r"proprietary information", r"trade secrets?", r"sensitive (?:business )?(?:information|data)", r"secret (?:business )?(?:information|materials)", r"nonpublic (?:business )?(?:information|data)", r"information (?:provided|disclosed|made available) by (?:either )?party"],
            "information_categories": [r"business.{0,35}(?:financial|technical|operational|customer)", r"(?:financial|technical|operational|customer).{0,35}information", r"business plans", r"customer lists", r"software.{0,20}(?:architecture|source code)"],
            "disclosure_formats": [r"written.{0,45}electronic.{0,45}oral", r"(?:written|electronic|oral) form", r"in (?:oral|written|electronic) form", r"whether disclosed orally, in writing or electronically", r"regardless of (?:the )?form(?:at)? of disclosure", r"in any (?:form|medium)", r"in whatever form", r"oral, visual, written, electronic or other form"],
            "confidentiality_boundary": [r"designated as confidential", r"marked.{0,30}confidential", r"identified.{0,30}confidential", r"at the sole discretion of", r"reasonably understood.{0,40}confidential", r"reasonably should be understood.{0,40}confidential", r"reasonably understood to be of a confidential nature", r"identified as confidential at the time of disclosure"],
        },
        "minimum_groups": 3,
        "required_groups": ["confidential_scope"],
        "red_flags": [
            ("all information whatsoever", r"all information whatsoever(?:.{0,70})?(?:confidential|secret|protected)?", "high"),
            ("sole discretion of disclosing party", r"sole discretion of the disclosing party", "high"),
            ("oral information excluded entirely", r"oral (?:information|disclosures?) (?:are|is) excluded", "medium"),
        ],
    },
    "B": {
        "heading": r"permitted use|use and purpose|purpose of disclosure",
        "groups": {
            "defined_business_purpose": [r"purpose of (?:evaluating|considering|pursuing|assessing)", r"(?:evaluate|evaluating|evaluation) (?:of )?(?:a )?(?:potential )?(?:business|commercial|vendor|strategic|joint)", r"permitted purpose", r"business relationship", r"potential transaction", r"proposed (?:transaction|relationship|project)"],
            "limited_use": [r"solely for (?:the )?(?:permitted |specific )?purpose", r"only for (?:the )?(?:permitted |specific )?purpose", r"exclusively for", r"shall (?:only|solely) use", r"limited to the purpose", r"for no other purpose", r"only in connection with.{0,80}(?:evaluation|assessment|transaction|relationship|project)", r"solely in connection with.{0,80}(?:evaluation|assessment|transaction|relationship|project)", r"(?:may|shall|must) only be used to.{0,80}(?:evaluate|assess|consider|pursue)", r"only be used for.{0,70}(?:evaluation|assessment|relationship|transaction|project)", r"shall not use.{0,70}(?:compete|competition|competitive|commercial benefit)", r"shall not exploit.{0,70}(?:information|materials|data)"],
            "prohibited_unrelated_use": [r"not use.{0,80}(?:other|unrelated|outside)", r"shall not.{0,60}(?:reverse engineer|exploit|commercially use)", r"prohibit.{0,60}(?:reverse engineering|unauthorized use)", r"no (?:commercial|competitive|unrelated) use"],
        },
        "minimum_groups": 2,
        "required_groups": ["defined_business_purpose"],
        "red_flags": [
            ("unrestricted use", r"(?:unrestricted|without restriction) use", "high"),
            ("any purpose", r"for any purpose", "high"),
            ("royalty-free license to use", r"royalty[- ]free licence? to use", "high"),
        ],
    },
    "C": {
        "heading": r"exclusions|exceptions|information not confidential",
        "groups": {
            "public_information_exception": [r"publicly available", r"public domain", r"becomes public.{0,40}(?:without|other than) (?:a )?breach", r"known to the public", r"generally available to the public", r"becomes generally known", r"enters the public domain through no act or omission", r"public knowledge through no breach"],
            "prior_knowledge_exception": [r"already known.{0,60}(?:prior to|before) disclosure", r"prior knowledge", r"in (?:the )?receiving party['’]?s possession.{0,60}before", r"lawfully known.{0,40}before disclosure", r"in the possession of the receiving party before receipt", r"documented in the recipient.{0,40}records before disclosure", r"known or available to the receiving party prior to"],
            "independent_development_exception": [r"independently developed", r"independent development", r"developed independently.{0,60}without (?:use of|reference to)", r"without reference to confidential information", r"created independently by the receiving party", r"developed without access to or use of the confidential information"],
            "third_party_source_exception": [r"lawfully obtained.{0,60}third party", r"received from a third party", r"third party.{0,50}authorized to disclose", r"rightfully received.{0,50}third party", r"received lawfully from a person not under an obligation of confidentiality", r"obtained from a source other than the disclosing party"],
        },
        "minimum_groups": 2,
        "required_groups": ["public_information_exception"],
        "red_flags": [
            ("no exclusions", r"no (?:exclusions|exceptions) (?:apply|shall apply)", "high"),
            ("regardless of public availability", r"regardless of (?:whether )?(?:the information is )?public(?:ly available)?", "high"),
        ],
    },
    "D": {
        "heading": r"confidentiality period|confidentiality duration|term of confidentiality|duration",
        "groups": {
            "stated_time_period": [r"\b\d{1,2}\s*\(?(?:one|two|three|four|five|six|seven|eight|nine|ten)?\)?\s*years?\b", r"\b(?:one|two|three|four|five)\s+years?\b", r"for a period of.{0,55}(?:years?|months?)", r"confidentiality period of", r"for (?:a )?(?:term|period) of (?:one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\s*(?:\(\s*\d+\s*\))?\s*years?", r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\s*\(\s*\d+\s*\)\s*years?"],
            "duration_anchor": [r"from the date of disclosure", r"following (?:the )?termination", r"after termination", r"after expiry", r"from (?:the )?effective date", r"from (?:the )?date (?:of|on) disclosure", r"from the last disclosure", r"following the last disclosure", r"after the expiration of this agreement", r"for so long as the information remains confidential"],
            "survival_of_obligation": [r"confidentiality.{0,70}shall survive", r"obligations? of confidentiality.{0,80}(?:remain|continue|survive)", r"shall remain in effect", r"survive any earlier termination", r"survival of confidentiality", r"confidentiality duties? (?:shall )?(?:continue|remain binding|remain in force)", r"recipient shall continue to protect", r"obligation to keep.{0,50}confidential shall continue"],
            "confidentiality_subject": [r"confidentiality obligations?", r"confidential information", r"strictly secret", r"obligations? of secrecy", r"non[- ]disclosure obligations?", r"confidentiality and non[- ]use", r"hold.{0,50}in confidence", r"keep.{0,40}confidential and not disclose"],
        },
        "minimum_groups": 3,
        "required_groups": ["confidentiality_subject"],
        "red_flags": [
            ("perpetual confidentiality", r"(?:confidentiality|confidential|secrecy|non[- ]disclosure).{0,100}(?:in perpetuity|perpetual(?:ly)?|indefinitely|forever|without any expiration date)|(?:in perpetuity|perpetual(?:ly)?|indefinitely|forever|without any expiration date).{0,100}(?:confidentiality|confidential|secrecy|non[- ]disclosure)", "high"),
            ("99-year confidentiality period", r"\b99\s*[- ]?years?\b", "high"),
        ],
    },
    "E": {
        "heading": r"return|destruction|deletion|disposition of information",
        "groups": {
            "return_or_destroy_action": [r"return.{0,55}(?:confidential|materials?|documents?|information|copies)", r"(?:destroy|delete|erase|securely dispose of).{0,65}(?:confidential|materials?|documents?|information|files?|copies)", r"(?:confidential|materials?|documents?|information|files?|copies).{0,65}(?:returned|destroyed|deleted|erased)", r"secure destruction", r"securely destroy", r"hand back.{0,50}(?:documents|materials|information|files)", r"deliver back.{0,50}(?:documents|materials|information|files)", r"dispose of.{0,60}(?:confidential|sensitive).{0,30}(?:data|materials|records)", r"render.{0,30}unrecoverable", r"permanently erase.{0,50}(?:data|files|information)"],
            "covered_copies_and_media": [r"copies", r"electronic files", r"all (?:copies|records|materials)", r"tangible documents", r"backup copies", r"derivative (?:materials|copies)", r"reproductions", r"extracts and summaries", r"notes derived from", r"all media containing"],
            "return_trigger": [r"upon (?:written )?request", r"upon termination", r"following termination", r"on expiration", r"at the end of (?:the )?agreement", r"within \d{1,3} days? of (?:termination|request)", r"within a reasonable period after (?:termination|request)", r"within \d{1,3} days?\b", r"upon expiry", r"following a request from the disclosing party"],
            "certification_or_confirmation": [r"written certification", r"certify (?:that )?(?:all )?(?:information|copies|materials).{0,40}(?:destroyed|deleted)", r"confirm.{0,40}(?:return|destruction|deletion)", r"certificate of destruction", r"written confirmation.{0,50}(?:returned|destroyed|deleted)", r"certify in writing.{0,60}(?:return|destruction|deletion)"],
        },
        "minimum_groups": 3,
        "required_groups": ["return_or_destroy_action", "return_trigger"],
        "red_flags": [
            ("may retain indefinitely", r"may retain.{0,50}(?:indefinitely|permanently|without time limit)", "high"),
            ("no obligation to return or destroy", r"no obligation to (?:return|destroy|delete)", "high"),
        ],
    },
    "F": {
        "heading": r"permitted disclosure|legal compulsion|disclosure required by law|disclosure to third parties|third[- ]party disclosure",
        "groups": {
            "permitted_representatives": [r"employees?.{0,80}(?:advisors?|representatives?|officers?)", r"legal (?:and |or )?financial advisors?", r"representatives.{0,70}(?:need to know|confidentiality obligations)", r"need[- ]to[- ]know basis", r"professional advisors?", r"affiliates?.{0,60}(?:employees|officers|advisers|representatives)", r"personnel and professional advisers", r"permitted recipients", r"persons who need to know"],
            "representative_safeguards": [r"bound by.{0,50}confidentiality obligations?", r"subject to.{0,50}(?:confidentiality|non[- ]disclosure)", r"written confidentiality obligations", r"at least as protective confidentiality obligations", r"bound by obligations no less protective", r"subject to duties of confidentiality equivalent to", r"ensure that each recipient complies"],
            "legally_compelled_disclosure": [r"compelled by law", r"required by (?:applicable )?law", r"court order", r"subpoena", r"governmental authority", r"regulatory (?:request|requirement|order)", r"legally required to disclose", r"required pursuant to applicable law", r"lawful process", r"legal process or governmental demand"],
            "notice_before_compelled_disclosure": [r"prior (?:written )?notice", r"prompt notice", r"notify.{0,50}before disclosure", r"where legally permitted.{0,60}notice", r"opportunity to seek a protective order", r"to the extent legally permitted.{0,60}notify", r"as soon as reasonably practicable.{0,50}notice", r"before producing the information"],
        },
        "minimum_groups": 3,
        "required_groups": ["permitted_representatives", "legally_compelled_disclosure"],
        "red_flags": [
            ("prohibited from disclosing to legal counsel", r"prohibit(?:ed|s)? from disclos(?:ing|ure).{0,60}(?:legal counsel|attorney|lawyer)", "high"),
            ("no notice required", r"no (?:prior )?notice (?:is )?required", "medium"),
            ("unrestricted third-party disclosure", r"(?:may|can|is entitled to) disclose.{0,140}(?:third parties|any third party).{0,100}(?:considers? (?:such )?disclosure useful|considers? it useful|at (?:its|their) discretion|for any purpose|useful for)", "high"),
        ],
    },
    "G": {
        "heading": r"security|safeguarding|protection of information|standard of care",
        "groups": {
            "standard_of_care": [r"reasonable care", r"same degree of care", r"no less than.{0,35}care", r"at least the same (?:degree|level) of care", r"industry[- ]standard.{0,35}safeguards?", r"same care it uses to protect its own.{0,60}(?:information|data)", r"commercially reasonable care", r"reasonable administrative, technical and physical safeguards"],
            "security_safeguards": [r"security measures", r"technical and organizational measures", r"administrative.{0,40}technical safeguards", r"safeguard(?:s|ing)? confidential", r"protect.{0,60}confidential information", r"appropriate technical and organizational measures", r"access controls", r"encryption.{0,60}(?:data|information)", r"maintain safeguards appropriate to the sensitivity"],
            "unauthorized_access_or_disclosure": [r"unauthorized access", r"unauthorized disclosure", r"prevent.{0,50}(?:access|disclosure|use) without authorization", r"against access.{0,40}unauthorized", r"theft or unauthorized disclosure", r"prevent unauthorized use", r"protect against unauthorized access, use or disclosure", r"prevent access by unauthorized persons"],
            "security_incident_notification": [r"notify.{0,50}(?:security incident|data breach|unauthorized access|breach)", r"prompt notification.{0,50}(?:breach|incident)", r"notify.{0,40}of any breach", r"security breach.{0,50}notice", r"notify the disclosing party promptly of any.{0,35}(?:incident|breach)", r"report any actual or suspected security incident", r"without undue delay.{0,50}(?:breach|incident)"],
        },
        "minimum_groups": 3,
        "required_groups": ["standard_of_care", "security_safeguards"],
        "red_flags": [
            ("no security obligation", r"no (?:security|safeguarding|protection) obligations?", "high"),
            ("strict liability for any cyberattack", r"strict liability.{0,50}(?:cyberattack|cyber attack|hacking|security incident)", "high"),
        ],
    },
    "H": {
        "heading": r"liability|remedies|indemnification|injunctive relief",
        "groups": {
            "breach_or_harm": [r"unauthorized disclosure.{0,80}(?:harm|damage|remedy)", r"breach.{0,80}(?:liability|damages|remedies)", r"irreparable harm", r"breach of this agreement", r"losses arising from"],
            "legal_remedies": [r"injunctive relief", r"equitable relief", r"specific performance", r"monetary damages", r"remedies available at law", r"seek an injunction", r"injunction or other equitable relief", r"equitable remedies", r"specific and injunctive relief", r"remedies shall be cumulative"],
            "liability_allocation": [r"liable for", r"liability of (?:either|each|the) party", r"indemnif(?:y|ies|ication)", r"damages arising from", r"each party shall be responsible"],
            "mutuality_or_limits": [r"either party", r"each party", r"both parties", r"mutual(?:ly)? liable", r"liability cap", r"limited to.{0,40}(?:fees|amount|sum)", r"aggregate liability shall not exceed", r"liability shall be capped at", r"maximum aggregate liability", r"subject to the liability cap", r"neither party.{0,80}liable for indirect"],
        },
        "minimum_groups": 2,
        "required_groups": ["legal_remedies"],
        "red_flags": [
            ("unlimited liability", r"(?:unlimited|uncapped|without (?:any )?limit(?:ation)?|no cap on|without (?:any )?cap).{0,65}(?:liability|losses|damages)|(?:liability|losses|damages).{0,65}(?:unlimited|uncapped|without (?:any )?limit(?:ation)?|without (?:any )?cap|no cap)", "high"),
            ("automatic liquidated damages", r"automatic(?:ally)? .{0,30}liquidated damages|liquidated damages (?:of|in the amount of).{0,35}(?:usd|\$|inr|eur|\d)", "high"),
            ("solely liable", r"solely liable", "high"),
            ("waives all defenses", r"waives? all defenses", "high"),
            ("broad liability without an express cap", r"(?:responsible|liable|indemnif(?:y|ies|ication)).{0,100}(?:all|any) (?:losses|claims|costs|damages).{0,90}(?:arising from|resulting from|caused by) (?:any )?breach", "high"),
        ],
    },
    "I": {
        "heading": r"governing law|jurisdiction|venue|dispute resolution",
        "groups": {
            "choice_of_law": [r"governed by.{0,50}laws of", r"laws of [A-Z][A-Za-z .'-]{1,45}", r"construed in accordance with.{0,50}law", r"governing law", r"law applicable to this agreement", r"laws of the jurisdiction of", r"governed by the laws of the state of", r"governed by the laws of the country of", r"interpreted under the laws of"],
            "forum_or_dispute_process": [r"courts? (?:located|of|in).{0,70}(?:jurisdiction|exclusive|shall have)", r"exclusive jurisdiction", r"submit to the jurisdiction", r"venue for (?:any )?(?:dispute|proceeding)", r"arbitration.{0,60}(?:rules|seat|administered)", r"dispute resolution", r"courts of competent jurisdiction in", r"submit to the exclusive jurisdiction of", r"venue shall be", r"exclusive venue"],
        },
        "minimum_groups": 2,
        "required_groups": ["choice_of_law"],
        "red_flags": [
            ("any court worldwide", r"(?:may sue|bring proceedings|bring an action).{0,50}any court worldwide|any court worldwide", "high"),
        ],
    },
    "J": {
        "heading": r"termination|expiration|expiry|term of agreement",
        "groups": {
            "termination_right": [r"either party may terminate", r"either party may end", r"may terminate.{0,50}agreement", r"terminate this agreement", r"termination upon.{0,40}notice", r"end this agreement", r"bring this agreement to an end", r"terminate the relationship by giving", r"may bring the agreement to an end", r"may terminate on written notice"],
            "notice_or_duration": [r"written notice", r"prior notice", r"\d{1,2} days?['’]? (?:prior )?notice", r"effective date of termination", r"term of this agreement"],
            "survival_after_termination": [r"shall survive termination", r"survive any termination", r"termination shall not relieve", r"obligations?.{0,80}survive.{0,40}termination", r"remain in effect after termination", r"provisions which by their nature should survive", r"termination shall not affect accrued rights", r"certain obligations will continue after expiry"],
        },
        "minimum_groups": 2,
        "required_groups": ["termination_right"],
        "red_flags": [
            ("cannot be terminated", r"cannot be terminated|may not be terminated|not capable of termination", "high"),
            ("only disclosing party may terminate", r"only the disclosing party may terminate", "high"),
        ],
    },
}


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def _sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    # Semicolons are useful clause boundaries in legal drafting; keep clause fragments.
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+(?=[A-Za-z0-9(“'])", clean) if part.strip()]


def _context_windows(block: dict, max_sentences: int = 4) -> list[str]:
    sentences = _sentences(block.get("text", ""))
    if not sentences:
        return []
    windows: list[str] = []
    for start in range(len(sentences)):
        for size in range(1, min(max_sentences, len(sentences) - start) + 1):
            windows.append(" ".join(sentences[start:start + size]))
    # Retain the entire section when compact: useful where a requirement is split across sentences.
    if len(block.get("text", "")) <= 1800:
        windows.append(block["text"])
    # Stable de-duplication.
    return list(dict.fromkeys(windows))



_HEADING_START = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*[.)]?)\s+[A-Z]|(?:ARTICLE|SECTION)\s+[A-Z0-9IVX]+\b)",
    re.IGNORECASE,
)


def _expanded_context_blocks(blocks: list[dict], max_blocks: int = 3, max_chars: int = 1800) -> list[dict]:
    """Join neighbouring short paragraphs on the same page when no new section starts.

    This allows a clause whose definition, obligation and exception are separated
    by paragraph breaks to be considered as a single local context, without
    merging across explicit numbered/ARTICLE/SECTION headings or page boundaries.
    """
    expanded = list(blocks)
    for start in range(len(blocks)):
        base = blocks[start]
        combined = str(base.get("text", "")).strip()
        for nxt in range(start + 1, min(len(blocks), start + max_blocks)):
            candidate = blocks[nxt]
            if candidate.get("page_number") != base.get("page_number"):
                break
            next_text = str(candidate.get("text", "")).strip()
            if not next_text or _HEADING_START.match(next_text):
                break
            if len(combined) + len(next_text) + 1 > max_chars:
                break
            combined = f"{combined} {next_text}"
            expanded.append({"page_number": base.get("page_number"), "text": combined, "combined_blocks": nxt - start + 1})
    return expanded

def _is_negated(text: str, start: int, end: int | None = None, matched_text: str = "") -> bool:
    """Recognise a red-flag phrase explicitly denied by nearby legal wording.

    The check is deliberately local: a negation elsewhere in a long section must
    not suppress a genuine red flag. Phrases such as "no liability cap" remain
    risk indicators and are not treated as negation.
    """
    end = start if end is None else end
    context = text[max(0, start - 90):min(len(text), max(end, start) + 70)].lower()
    target = (matched_text or text[start:end]).lower()
    danger = r"(?:perpetual(?:ly)?|in perpetuity|indefinite(?:ly)?|forever|unlimited|uncapped)"
    # Direct denial: "shall not be perpetual", "is not unlimited", etc.
    if re.search(rf"\b(?:not|never|isn't|aren't|is not|are not|shall not|will not|must not|cannot be)\b(?:\s+\w+){{0,3}}\s+{danger}\b", context):
        return True
    if re.search(rf"\b{danger}\b", target) and re.search(r"\b(?:not|never|isn't|aren't|shall not|will not|must not)\b", context):
        # Only suppress when the negation and target are close in the same clause.
        neg = re.search(r"\b(?:not|never|isn't|aren't|shall not|will not|must not)\b", context)
        pos = re.search(danger, context)
        if neg and pos and abs(pos.start() - neg.end()) <= 45:
            return True
    return False


def _find_red_flags(clause_id: str, window: str, standard_flags: list[str]) -> list[dict]:
    rule = RULES.get(clause_id, {})
    found: list[dict] = []
    for label, pattern, severity in rule.get("red_flags", []):
        for match in re.finditer(pattern, window, re.IGNORECASE):
            if not _is_negated(window, match.start(), match.end(), match.group(0)):
                found.append({"label": label, "severity": severity, "matched_text": match.group(0)[:180], "source": "clause_rule"})
                break
    # Also honour the source JSON red flags, but require a real local phrase match.
    lower = window.lower()
    for flag in standard_flags:
        normalized = str(flag).strip().lower()
        if len(normalized) < 12 or len(normalized.split()) < 2 or normalized in {item["label"] for item in found}:
            continue
        if normalized in lower and not _is_negated(window, lower.find(normalized), lower.find(normalized) + len(normalized), normalized):
            if not any(normalized in item["matched_text"].lower() for item in found):
                found.append({"label": normalized, "severity": "high" if any(x in normalized for x in ("perpetual", "unlimited", "sole discretion", "automatic")) else "medium", "matched_text": normalized, "source": "baseline_indicator"})
    return found


def _group_matches(clause_id: str, window: str) -> dict[str, str]:
    rule = RULES.get(clause_id, {})
    results: dict[str, str] = {}
    for group, patterns in rule.get("groups", {}).items():
        for pattern in patterns:
            match = re.search(pattern, window, re.IGNORECASE)
            if match:
                results[group] = match.group(0)[:160]
                break
    return results


def _heading_bonus(clause_id: str, text: str) -> float:
    pattern = RULES.get(clause_id, {}).get("heading", "")
    if pattern and re.search(pattern, text[:180], re.IGNORECASE):
        return 0.18
    return 0.0


def _score_windows(clause_id: str, blocks: list[dict], standard_flags: list[str]) -> dict:
    best: dict[str, Any] = {"score": 0.0, "block": None, "window": "", "groups": {}, "flags": []}
    for block in _expanded_context_blocks(blocks):
        for window in _context_windows(block):
            groups = _group_matches(clause_id, window)
            heading_hint = _heading_bonus(clause_id, block.get("text", ""))
            flags = _find_red_flags(clause_id, window, standard_flags)
            # Do not attribute a generic red-flag phrase to a different clause topic.
            # Clause-specific patterns (e.g. "unlimited liability") are self-contextual;
            # baseline phrases require either a matching concept or the correct section.
            if flags and not groups and not heading_hint:
                flags = []
            denominator = max(len(RULES.get(clause_id, {}).get("groups", {})), 1)
            coverage = len(groups) / denominator
            score = coverage + heading_hint
            # Red flags increase evidence relevance but do not count as a positive clause concept.
            if flags:
                score += 0.12
            if score > best["score"]:
                best = {"score": score, "block": block, "window": window, "groups": groups, "flags": flags, "coverage": coverage}
    return best


def analyze_offline(contract_text: str, standards_data: dict, evidence_blocks: list[dict]) -> dict:
    """Run contextual rule screening plus a local learned classifier for all baseline clauses."""
    standards = standards_data.get("clauses", [])
    clause_results = []
    # The learned classifier is a conservative second opinion. It only routes
    # high-confidence, unambiguous matches to review; it never auto-approves.
    learned_by_clause: dict[str, dict[str, Any]] = {}
    for block in evidence_blocks:
        prediction = predict_clause(block.get("text", ""))
        if prediction:
            cid = prediction["clause_id"]
            current = learned_by_clause.get(cid)
            if current is None or prediction["confidence"] > current["prediction"]["confidence"]:
                learned_by_clause[cid] = {"prediction": prediction, "block": block}

    for standard in standards:
        cid = str(standard.get("id", "")).upper()
        cname = str(standard.get("name", cid))
        best = _score_windows(cid, evidence_blocks, standard.get("red_flag_indicators", []))
        learned_entry = learned_by_clause.get(cid)
        learned_match = learned_entry["prediction"] if learned_entry else None
        group_hits = best.get("groups", {})
        flags = best.get("flags", [])
        all_groups = list(RULES.get(cid, {}).get("groups", {}))
        required_groups = RULES.get(cid, {}).get("required_groups", [])
        coverage = len(group_hits) / max(len(all_groups), 1)
        required_present = all(group in group_hits for group in required_groups)
        minimum_groups = int(RULES.get(cid, {}).get("minimum_groups", max(1, len(all_groups) // 2 + 1)))
        enough_evidence = len(group_hits) >= minimum_groups and required_present

        # High-confidence concept groups, rather than isolated common words, drive status.
        if (not group_hits or not required_present) and not flags:
            # An ML-only signal is never enough to claim that a clause is present.
            # It prevents a likely paraphrase from being silently called missing.
            status = "Needs Review" if learned_match else "Missing / Not Detected"
        elif flags or not enough_evidence:
            status = "Needs Review"
        else:
            status = "Present"

        severity_values = [item.get("severity", "medium") for item in flags]
        if any(value == "high" for value in severity_values):
            flag_strength = 1.0
        elif flags:
            flag_strength = 0.55
        else:
            flag_strength = 0.0

        # Calibrate concept coverage for the existing fuzzy system. Full, multi-concept
        # coverage maps to its high-coverage region; partial matches stay uncertain.
        fuzzy_coverage = coverage if coverage else 0.0
        fuzzy = infer_clause_risk(fuzzy_coverage, flag_strength)

        if status == "Missing / Not Detected":
            risk = "High Risk"
        elif any(value == "high" for value in severity_values):
            risk = "High Risk"
        elif flags:
            risk = "Medium Risk"
        elif learned_match and (not group_hits or not required_present):
            risk = "Medium Risk"
        elif status == "Needs Review":
            risk = "High Risk" if coverage < 0.25 else "Medium Risk"
        else:
            risk = fuzzy["fuzzy_risk_level"]

        # Validate the final status/risk pair with the CSP consistency constraints.
        csp = validate_clause_assignment(status, risk)
        if not csp.get("csp_solved"):
            status, risk = "Needs Review", "Medium Risk"
            csp = validate_clause_assignment(status, risk)

        block = best.get("block") or (learned_entry.get("block") if learned_entry else None)
        evidence_page = block.get("page_number") if block else None
        evidence_window = best.get("window") or (block.get("text", "") if learned_match and block else "")
        if block and evidence_window:
            evidence = f"Page {evidence_page}: {evidence_window}"
            evidence = evidence[:517].rstrip() + "..." if len(evidence) > 520 else evidence
            verified = True
            verification_note = "Evidence is an exact text window extracted from the source PDF section."
        else:
            evidence = "No sufficiently specific clause evidence detected by contextual offline screening."
            verified = None
            verification_note = "No supporting clause excerpt was identified; manual review is required."

        matched_keywords = sorted({
            str(keyword).lower() for keyword in standard.get("keywords", [])
            if keyword and best.get("window") and str(keyword).lower() in best["window"].lower()
        })
        matched_red_flags = sorted({item["label"] for item in flags})
        matched_concepts = [
            {"concept": group, "matched_text": matched_text}
            for group, matched_text in group_hits.items()
        ]
        missing_concepts = [group for group in all_groups if group not in group_hits]

        if status == "Missing / Not Detected":
            difference = f"No sufficiently specific evidence for the {cname} concepts was detected. Generic words alone are not treated as proof that a clause exists."
            implication = "If an equivalent provision is genuinely absent, the agreement may leave a protective gap on this topic."
            action = f"Check the complete NDA manually for an equivalent {cname} provision, including differently worded sections."
        elif flags:
            flag_text = ", ".join(matched_red_flags)
            difference = f"The clause topic was detected, but contextual red-flag pattern(s) were found: {flag_text}. This is a screening signal, not a legal conclusion."
            implication = "The wording may be broad, one-sided or unusually restrictive and should be reviewed in context."
            action = "Review the quoted wording, its exceptions and commercial context with an authorised legal reviewer."
        elif status == "Present":
            difference = f"Detected {len(group_hits)} of {len(all_groups)} clause concept groups ({coverage:.0%} concept coverage) in a locally relevant section/window. This does not establish legal sufficiency."
            implication = "The main baseline concepts appear to be addressed; a human should still review the complete provision."
            action = "Verify the full wording and compare it with the organisation's approved template."
        else:
            difference = f"Partial evidence was found for {cname} ({coverage:.0%} concept coverage); missing concepts: {', '.join(missing_concepts) or 'none'}. The provision may be incomplete or expressed differently."
            implication = "Some relevant ideas appear, but the available text does not clearly establish all expected obligations."
            action = "Send this clause for human review and confirm the missing concepts in the original agreement."

        clause_results.append({
            "clause_id": cid,
            "clause_name": cname,
            "status": status,
            "risk_level": risk,
            "extracted_evidence": evidence,
            "evidence_page": evidence_page,
            "evidence_verified": verified,
            "verification_note": verification_note,
            "standard_baseline": standard.get("standard_description", ""),
            "expected_purpose": standard.get("expected_purpose", ""),
            "difference_explanation": difference,
            "potential_implication": implication,
            "suggested_action": action,
            "matched_keywords": matched_keywords,
            "matched_red_flags": matched_red_flags,
            "matched_concepts": matched_concepts,
            "missing_concepts": missing_concepts,
            "concept_coverage": round(coverage, 3),
            "coverage_ratio": round(fuzzy_coverage, 3),
            "raw_concept_coverage": round(coverage, 3),
            "red_flag_strength": round(flag_strength, 3),
            "fuzzy_risk_score": fuzzy["fuzzy_risk_score"],
            "fuzzy_risk_level": fuzzy["fuzzy_risk_level"],
            "fuzzy_memberships": fuzzy["membership_degrees"],
            "csp_solved": csp.get("csp_solved", True),
            "csp_note": csp.get("csp_note", "") + " Clause status was determined by local concept evidence and red-flag rules.",
            "offline_method": "Contextual rule matching plus a locally trained TF-IDF word/character n-gram classifier; ML-only signals are routed to review, never auto-approved.",
            "offline_confidence": round(learned_match["confidence"] if learned_match and not group_hits else min(0.99, 0.25 + 0.75 * coverage + (0.10 if block else 0)), 3),
            "learned_match": learned_match,
            "matched_section_page": evidence_page,
        })

    return {"clauses": clause_results, "engine_name": "Hybrid Local NLP + Learned Clause Classifier", "engine_version": "4.0"}
