"""A compact AC-3 + backtracking CSP solver used to reconcile clause labels."""
from __future__ import annotations

from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Tuple

Constraint = Tuple[str, str, Callable[[object, object], bool]]


def solve_csp(domains: Dict[str, List[object]], constraints: Iterable[Constraint]) -> Optional[Dict[str, object]]:
    """Solve a finite-domain CSP with AC-3, MRV, degree tie-break and backtracking."""
    constraints = list(constraints)
    neighbors: Dict[str, List[str]] = {v: [] for v in domains}
    predicates: Dict[Tuple[str, str], List[Callable[[object, object], bool]]] = {}
    for left, right, predicate in constraints:
        if left not in domains or right not in domains:
            raise ValueError(f"Constraint references unknown variable: {left}, {right}")
        neighbors[left].append(right)
        neighbors[right].append(left)
        predicates.setdefault((left, right), []).append(predicate)
        predicates.setdefault((right, left), []).append(lambda b, a, p=predicate: p(a, b))

    initial = {key: list(values) for key, values in domains.items()}
    if any(not values for values in initial.values()):
        return None

    def consistent(left: str, left_value: object, right: str, right_value: object) -> bool:
        return all(p(left_value, right_value) for p in predicates.get((left, right), []))

    def revise(current: Dict[str, List[object]], left: str, right: str) -> bool:
        revised = False
        kept = [
            value for value in current[left]
            if any(consistent(left, value, right, other) for other in current[right])
        ]
        if len(kept) != len(current[left]):
            current[left] = kept
            revised = True
        return revised

    def ac3(current: Dict[str, List[object]], queue_items=None) -> bool:
        queue = deque(queue_items or [(x, y) for x in current for y in neighbors[x]])
        while queue:
            left, right = queue.popleft()
            if revise(current, left, right):
                if not current[left]:
                    return False
                for other in neighbors[left]:
                    if other != right:
                        queue.append((other, left))
        return True

    if not ac3(initial):
        return None

    def backtrack(current: Dict[str, List[object]]) -> Optional[Dict[str, object]]:
        unassigned = [v for v, values in current.items() if len(values) > 1]
        if not unassigned:
            return {v: values[0] for v, values in current.items()}
        # Minimum Remaining Values; degree heuristic breaks ties.
        variable = min(unassigned, key=lambda v: (len(current[v]), -len(neighbors[v])))
        for value in current[variable]:
            candidate = {v: list(values) for v, values in current.items()}
            candidate[variable] = [value]
            if ac3(candidate, [(n, variable) for n in neighbors[variable]]):
                result = backtrack(candidate)
                if result is not None:
                    return result
        return None

    return backtrack(initial)


def reconcile_clause_labels(signals: dict, fuzzy_risk_level: str) -> dict:
    """Model a clause's status and risk as finite-domain CSP variables."""
    clause_id = signals.get("clause_id", "?")
    coverage = float(signals.get("coverage_ratio", 0.0))
    red_flags = list(signals.get("matched_red_flags", []))

    if coverage <= 0.03:
        status_domain = ["Missing / Not Detected"]
    elif red_flags:
        status_domain = ["Needs Review"]
    elif fuzzy_risk_level == "High Risk":
        status_domain = ["Needs Review", "Missing / Not Detected"] if coverage < 0.25 else ["Needs Review"]
    elif fuzzy_risk_level == "Medium Risk":
        status_domain = ["Needs Review", "Present"] if coverage >= 0.25 else ["Needs Review"]
    else:
        status_domain = ["Present", "Needs Review"] if coverage >= 0.25 else ["Needs Review"]

    risk_domain = {
        "Low Risk": ["Low Risk", "Medium Risk"],
        "Medium Risk": ["Medium Risk", "High Risk"],
        "High Risk": ["High Risk"],
    }.get(fuzzy_risk_level, ["Medium Risk"])
    domains = {f"{clause_id}.status": status_domain, f"{clause_id}.risk": risk_domain}
    compatible = {
        "Present": {"Low Risk", "Medium Risk"},
        "Needs Review": {"Medium Risk", "High Risk"},
        "Missing / Not Detected": {"High Risk"},
    }
    constraints = [
        (f"{clause_id}.status", f"{clause_id}.risk", lambda status, risk: risk in compatible.get(status, set())),
    ]
    if red_flags:
        status_domain = [status for status in status_domain if status != "Present"]
        domains[f"{clause_id}.status"] = status_domain
    if coverage >= 0.25:
        domains[f"{clause_id}.status"] = [status for status in domains[f"{clause_id}.status"] if status != "Missing / Not Detected"]
    solution = solve_csp(domains, constraints)
    if solution is None:
        return {"status": "Needs Review", "risk_level": "Medium Risk", "csp_solved": False,
                "csp_note": "No fully consistent assignment was found; conservative review labels were applied."}
    return {
        "status": solution[f"{clause_id}.status"],
        "risk_level": solution[f"{clause_id}.risk"],
        "csp_solved": True,
        "csp_note": "Status and risk labels satisfy the clause consistency constraints (AC-3 + backtracking).",
    }


def validate_clause_assignment(status: str, risk_level: str) -> dict:
    """Run the same CSP constraints against one externally proposed assignment."""
    compatible = {
        "Present": {"Low Risk", "Medium Risk"},
        "Needs Review": {"Medium Risk", "High Risk"},
        "Missing / Not Detected": {"High Risk"},
    }
    domains = {"status": [status], "risk": [risk_level]}
    solution = solve_csp(domains, [
        ("status", "risk", lambda status_value, risk_value: risk_value in compatible.get(status_value, set()))
    ])
    return {
        "csp_solved": solution is not None,
        "csp_note": "AI-proposed status/risk pair satisfies CSP constraints." if solution is not None
                    else "AI-proposed status/risk pair violates the defined consistency constraints; human review is required.",
    }
