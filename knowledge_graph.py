"""Build a Graphviz representation from the structured NDA reference library."""
from __future__ import annotations

import json


def build_nda_knowledge_graph(standards_data: dict) -> dict:
    """Create nodes and typed edges from the project's structured clause knowledge."""
    nodes = [{"id": "NDA", "label": "Mutual NDA", "type": "document"}]
    edges = []
    for clause in standards_data.get("clauses", []):
        cid = str(clause["id"])
        clause_node = f"CLAUSE_{cid}"
        nodes.append({"id": clause_node, "label": f"{cid}. {clause['name']}", "type": "clause"})
        edges.append({"source": "NDA", "target": clause_node, "relation": "contains"})
        for index, flag in enumerate(clause.get("red_flag_indicators", []), start=1):
            flag_node = f"FLAG_{cid}_{index}"
            nodes.append({"id": flag_node, "label": str(flag), "type": "risk_indicator"})
            edges.append({"source": clause_node, "target": flag_node, "relation": "may contain risk indicator"})
        purpose_node = f"PURPOSE_{cid}"
        purpose = clause.get("expected_purpose", "")
        if purpose:
            nodes.append({"id": purpose_node, "label": purpose, "type": "purpose"})
            edges.append({"source": clause_node, "target": purpose_node, "relation": "supports"})
    return {"nodes": nodes, "edges": edges}


def to_graphviz_dot(graph: dict) -> str:
    """Return safe DOT source; all labels are JSON-escaped string literals."""
    type_styles = {
        "document": ("#DCEBFF", "#1D4ED8", "box"),
        "clause": ("#E8F2EC", "#166534", "ellipse"),
        "risk_indicator": ("#FEE2E2", "#991B1B", "note"),
        "purpose": ("#F1F5F9", "#475569", "box"),
    }
    lines = ["digraph CIRD_Knowledge {", '  graph [rankdir="LR", bgcolor="transparent", nodesep="0.35", ranksep="0.65"];',
             '  node [fontname="Arial", fontsize="9", style="filled,rounded", margin="0.10,0.06"];',
             '  edge [fontname="Arial", fontsize="8", color="#64748B", fontcolor="#475569"];']
    for node in graph.get("nodes", []):
        fill, outline, shape = type_styles.get(node.get("type"), ("#FFFFFF", "#64748B", "box"))
        label = json.dumps(str(node.get("label", "")), ensure_ascii=False)
        node_id = json.dumps(str(node.get("id", "")))
        lines.append(f"  {node_id} [label={label}, shape={shape}, fillcolor=\"{fill}\", color=\"{outline}\"];")
    for edge in graph.get("edges", []):
        src, dst = json.dumps(str(edge["source"])), json.dumps(str(edge["target"]))
        label = json.dumps(str(edge.get("relation", "")), ensure_ascii=False)
        lines.append(f"  {src} -> {dst} [label={label}];")
    lines.append("}")
    return "\n".join(lines)
