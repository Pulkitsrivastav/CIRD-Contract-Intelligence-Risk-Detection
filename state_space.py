"""Breadth-first workflow-state search for the offline contract-analysis pipeline."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional


WORKFLOW_GRAPH = {
    "START": ["PDF_VALIDATED"],
    "PDF_VALIDATED": ["TEXT_EXTRACTED"],
    "TEXT_EXTRACTED": ["OFFLINE_ANALYSIS"],
    "OFFLINE_ANALYSIS": ["ANALYSIS_READY"],
    "ANALYSIS_READY": ["REPORT_READY"],
    "REPORT_READY": [],
}

STATE_LABELS = {
    "START": "Initial state: no document processed",
    "PDF_VALIDATED": "PDF validated",
    "TEXT_EXTRACTED": "Contract text extracted",
    "OFFLINE_ANALYSIS": "Local contextual NLP and rule-based analysis",
    "ANALYSIS_READY": "Clause analysis and risk assessment complete",
    "REPORT_READY": "Goal state: report is ready",
}


def breadth_first_workflow_path(start: str = "START", goal: str = "REPORT_READY",
                                available_transitions: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Return the shortest workflow path using BFS over the state-transition graph."""
    graph = available_transitions or WORKFLOW_GRAPH
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path
        for next_state in graph.get(state, []):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state]))
    return []


def plan_analysis_route() -> List[str]:
    """Return the only supported processing route: local offline analysis."""
    return breadth_first_workflow_path()
