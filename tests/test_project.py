import json
import unittest
from pathlib import Path

from analyzer import (
    _split_into_evidence_blocks,
    analyze_contract,
    analyze_with_keywords,
    generate_json_report,
    generate_markdown_report,
    load_standard_clauses,
)
from csp_solver import reconcile_clause_labels, solve_csp, validate_clause_assignment
from document_utils import extract_text_from_pdf
from evaluation import run_sample_evaluation
from fuzzy_logic import infer_clause_risk
from knowledge_graph import build_nda_knowledge_graph, to_graphviz_dot
from state_space import plan_analysis_route
from review import apply_human_reviews
from review_store import (
    load_review_store, save_review_store, build_review_candidates, _extract_additional_sections,
    approve_permanently, semi_approve, promote_pending_to_baseline,
    remove_pending, reject_remove, apply_approved_baseline_matches,
)

ROOT = Path(__file__).resolve().parents[1]


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.standards = load_standard_clauses(str(ROOT / "data" / "nda_standards.json"))

    def _analyze_sample(self, filename):
        path = ROOT / "sample_documents" / filename
        extracted = extract_text_from_pdf(path.read_bytes(), filename)
        self.assertTrue(extracted["success"], extracted.get("error"))
        return extracted, analyze_with_keywords(extracted["full_text"], self.standards)

    def test_standard_sample_is_low_risk_and_all_topics_present(self):
        _, result = self._analyze_sample("standard_mutual_nda.pdf")
        self.assertEqual(result["score_summary"]["overall_category"], "Low Risk")
        self.assertEqual(result["score_summary"]["clauses_reviewed"], 10)
        self.assertTrue(all(c["status"] == "Present" for c in result["clauses"]))
        self.assertTrue(all(c["evidence_verified"] for c in result["clauses"]))

    def test_flawed_sample_detects_critical_risks_and_omissions(self):
        _, result = self._analyze_sample("high_risk_flawed_nda.pdf")
        by_id = {c["clause_id"]: c for c in result["clauses"]}
        self.assertEqual(result["score_summary"]["overall_category"], "High Risk")
        self.assertEqual(by_id["D"]["risk_level"], "High Risk")
        self.assertEqual(by_id["H"]["risk_level"], "High Risk")
        for cid in ("C", "E", "F", "G"):
            self.assertEqual(by_id[cid]["status"], "Missing / Not Detected")
        self.assertIn("perpetual", " ".join(by_id["D"]["matched_red_flags"]))
        self.assertIn("unlimited liability", " ".join(by_id["H"]["matched_red_flags"]))
        self.assertIn("Confidentiality Duration", by_id["D"]["extracted_evidence"])
        self.assertIn("Remedies and Liability", by_id["H"]["extracted_evidence"])
        self.assertNotEqual(by_id["D"]["extracted_evidence"], by_id["H"]["extracted_evidence"])

    def test_evidence_blocks_split_page_and_numbered_sections(self):
        text = "--- Page 1 ---\n1. Definition\nConfidential information includes technical data.\n2. Duration\nObligations survive for three years.\n--- Page 2 ---\n1. Other\nAnother provision is included."
        blocks = _split_into_evidence_blocks(text)
        self.assertGreaterEqual(len(blocks), 3)
        self.assertTrue(any("Confidential information" in b["text"] for b in blocks))
        self.assertTrue(any(b["page_number"] == 2 for b in blocks))

    def test_fuzzy_risk_increases_for_low_coverage_and_red_flags(self):
        low = infer_clause_risk(0.9, 0.0)["fuzzy_risk_score"]
        high = infer_clause_risk(0.05, 1.0)["fuzzy_risk_score"]
        self.assertLess(low, high)

    def test_csp_solver_handles_map_coloring(self):
        colors = ["red", "green", "blue"]
        domains = {name: colors[:] for name in "ABC"}
        constraints = [
            ("A", "B", lambda a, b: a != b),
            ("B", "C", lambda b, c: b != c),
            ("A", "C", lambda a, c: a != c),
        ]
        solution = solve_csp(domains, constraints)
        self.assertIsNotNone(solution)
        self.assertEqual(len(set(solution.values())), 3)

    def test_clause_csp_reconciles_valid_status_and_risk(self):
        result = reconcile_clause_labels({"clause_id": "D", "coverage_ratio": 0.1, "matched_red_flags": ["perpetual"]}, "High Risk")
        self.assertTrue(result["csp_solved"])
        self.assertEqual(result["status"], "Needs Review")
        self.assertEqual(result["risk_level"], "High Risk")

    def test_csp_rejects_inconsistent_status_risk_pairs(self):
        self.assertTrue(validate_clause_assignment("Present", "Medium Risk")["csp_solved"])
        self.assertFalse(validate_clause_assignment("Present", "High Risk")["csp_solved"])
        self.assertFalse(validate_clause_assignment("Missing / Not Detected", "Low Risk")["csp_solved"])


    def test_offline_result_has_no_external_service_metadata(self):
        result = analyze_contract("Example contract text", self.standards)
        self.assertNotIn("provider", result)
        self.assertNotIn("model", result)
        self.assertNotIn("fallback_notice", result)
        self.assertEqual(result["method_details"]["state_space_path"], plan_analysis_route())

    def test_state_space_bfs_has_only_the_offline_path(self):
        path = plan_analysis_route()
        self.assertEqual(path, ["START", "PDF_VALIDATED", "TEXT_EXTRACTED", "OFFLINE_ANALYSIS", "ANALYSIS_READY", "REPORT_READY"])

    def test_knowledge_graph_has_all_clause_nodes_and_edges(self):
        graph = build_nda_knowledge_graph(self.standards)
        clause_nodes = [node for node in graph["nodes"] if node["type"] == "clause"]
        self.assertEqual(len(clause_nodes), 10)
        dot = to_graphviz_dot(graph)
        self.assertIn("digraph CIRD_Knowledge", dot)
        self.assertIn("Confidentiality Period", dot)

    def test_pdf_extraction_rejects_empty_non_pdf_and_image_only(self):
        self.assertFalse(extract_text_from_pdf(b"", "empty.pdf")["success"])
        self.assertFalse(extract_text_from_pdf(b"not a pdf", "bad.pdf")["success"])
        import fitz
        image_only = fitz.open()
        image_only.new_page(width=300, height=300)
        scanned_bytes = image_only.tobytes()
        image_only.close()
        scanned_result = extract_text_from_pdf(scanned_bytes, "scanned.pdf")
        self.assertFalse(scanned_result["success"])
        self.assertIn("OCR", scanned_result["error"])

    def test_reports_are_valid_and_contain_methodology(self):
        extracted, result = self._analyze_sample("standard_mutual_nda.pdf")
        md = generate_markdown_report(result, extracted)
        payload = json.loads(generate_json_report(result, extracted))
        self.assertIn("Mamdani", md)
        self.assertIn("method_details", payload)
        self.assertEqual(len(payload["clause_comparisons"]), 10)
        result["approval_workflow_summary"] = {
            "auto_approved_count": 8,
            "awaiting_human_approval_count": 2,
            "semi_approved_queue_count": 1,
            "permanent_supplemental_baseline_count": 3,
        }
        approval_md = generate_markdown_report(result, extracted)
        self.assertIn("Human Approval Workflow Summary", approval_md)
        self.assertIn("**Permanent supplemental baseline examples:** 3", approval_md)

    def test_analyze_contract_always_uses_offline_engine(self):
        result = analyze_contract("Example contract text", self.standards)
        self.assertTrue(result["success"])
        self.assertIn("Offline-Only", result["analysis_mode"])
        self.assertNotIn("fallback_notice", result)
        self.assertEqual(result["method_details"]["state_space_path"], plan_analysis_route())

    def test_human_review_updates_status_score_and_audit_trail(self):
        _, result = self._analyze_sample("standard_mutual_nda.pdf")
        reviewed = apply_human_reviews(result, {
            "D": {"status": "Needs Review", "risk_level": "High Risk", "note": "Manual reviewer found an unusual exception."}
        }, self.standards["clauses"])
        clause = next(item for item in reviewed["clauses"] if item["clause_id"] == "D")
        self.assertTrue(clause["human_reviewed"])
        self.assertEqual(clause["pre_review_status"], "Present")
        self.assertEqual(clause["risk_level"], "High Risk")
        self.assertIn("Manual reviewer", clause["human_review_note"])
        self.assertEqual(clause["fuzzy_risk_score"], clause["engine_fuzzy_risk_score"])
        self.assertEqual(clause["human_review_score"], 90.0)
        self.assertIn("human-reviewed clauses use", reviewed["score_summary"]["formula_explanation"].lower())
        self.assertGreater(reviewed["score_summary"]["overall_score"], result["score_summary"]["overall_score"])
        markdown = generate_markdown_report(reviewed, self._analyze_sample("standard_mutual_nda.pdf")[0])
        self.assertIn("Human-Reviewed Category Override", markdown)
        self.assertIn("Manual reviewer", markdown)
        self.assertEqual(result["clauses"][3]["status"], "Present")  # original remains unchanged










    def test_article_or_section_heading_is_detected_as_an_additional_provision(self):
        text = (
            "--- Page 1 ---\n"
            "ARTICLE IV - Audit and Inspection Rights\n"
            "The disclosing party may conduct a reasonable security audit once per year with prior notice. "
            "The receiving party will cooperate and address material findings within a reasonable period."
        )
        sections = _extract_additional_sections(text, self.standards["clauses"])
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["title"], "Audit and Inspection Rights")
        self.assertIn("reasonable security audit", sections[0]["evidence"])

    def test_new_numbered_provision_is_sent_for_human_approval(self):
        extracted, result = self._analyze_sample("standard_mutual_nda.pdf")
        workflow = build_review_candidates(
            result, extracted["full_text"], "doc-hash", extracted["filename"],
            load_review_store(ROOT / "data" / "__nonexistent_test_store__.json"), self.standards["clauses"],
        )
        titles = [item["title"] for item in workflow["candidates"]]
        self.assertIn("Parties and Effective Date", titles)
        self.assertEqual(len(workflow["auto_approved"]), 10)

    def test_permanently_approved_additional_section_matches_future_document(self):
        from tempfile import TemporaryDirectory
        text = (
            "--- Page 1 ---\n"
            "ARTICLE IV - Audit and Inspection Rights\n"
            "The disclosing party may conduct a reasonable security audit once per year with prior notice. "
            "The receiving party will cooperate and address material findings within a reasonable period."
        )
        empty_analysis = {"clauses": []}
        with TemporaryDirectory() as temp:
            path = Path(temp) / "human_approval_store.json"
            store = load_review_store(path)
            first = build_review_candidates(empty_analysis, text, "first-doc", "first.pdf", store, self.standards["clauses"])
            self.assertEqual(len(first["candidates"]), 1)
            candidate = first["candidates"][0]
            approve_permanently(candidate, "Low Risk", store, path)
            reloaded = load_review_store(path)
            second = build_review_candidates(empty_analysis, text, "second-doc", "second.pdf", reloaded, self.standards["clauses"])
            self.assertEqual(len(second["candidates"]), 0)
            self.assertEqual(len(second["recognized_additional"]), 1)
            self.assertEqual(second["recognized_additional"][0]["title"], "Audit and Inspection Rights")

    def test_permanent_approval_persists_and_matches_future_contract(self):
        from tempfile import TemporaryDirectory
        extracted, result = self._analyze_sample("high_risk_flawed_nda.pdf")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "human_approval_store.json"
            store = load_review_store(path)
            workflow = build_review_candidates(result, extracted["full_text"], "flawed-hash", extracted["filename"], store, self.standards["clauses"])
            candidate = next(item for item in workflow["candidates"] if item.get("clause_id") == "D")
            approve_permanently(candidate, "Medium Risk", store, path)
            reloaded = load_review_store(path)
            self.assertEqual(len(reloaded["approved_baseline"]), 1)
            matched = apply_approved_baseline_matches(result, reloaded, self.standards["clauses"])
            duration = next(item for item in matched["clauses"] if item["clause_id"] == "D")
            self.assertTrue(duration.get("baseline_approved_match"))
            self.assertEqual(duration["risk_level"], "Medium Risk")
            new_workflow = build_review_candidates(matched, extracted["full_text"], "flawed-hash", extracted["filename"], reloaded, self.standards["clauses"])
            self.assertNotIn("D", [item.get("clause_id") for item in new_workflow["candidates"]])

    def test_semi_approval_is_queued_not_added_to_baseline(self):
        from tempfile import TemporaryDirectory
        extracted, result = self._analyze_sample("high_risk_flawed_nda.pdf")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "human_approval_store.json"
            store = load_review_store(path)
            workflow = build_review_candidates(result, extracted["full_text"], "flawed-hash", extracted["filename"], store, self.standards["clauses"])
            candidate = next(item for item in workflow["candidates"] if item.get("clause_id") == "H")
            semi_approve(candidate, store, path)
            reloaded = load_review_store(path)
            self.assertEqual(len(reloaded["pending_approvals"]), 1)
            self.assertEqual(len(reloaded["approved_baseline"]), 0)
            self.assertEqual(reloaded["pending_approvals"][0]["queue_status"], "Semi-approved — awaiting baseline decision")

    def test_pending_item_can_be_promoted_or_removed_without_cross_effects(self):
        from tempfile import TemporaryDirectory
        extracted, result = self._analyze_sample("high_risk_flawed_nda.pdf")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "human_approval_store.json"
            store = load_review_store(path)
            workflow = build_review_candidates(result, extracted["full_text"], "flawed-hash", extracted["filename"], store, self.standards["clauses"])
            candidate = next(item for item in workflow["candidates"] if item.get("clause_id") == "D")
            semi_approve(candidate, store, path)
            promoted = promote_pending_to_baseline(candidate["id"], "Medium Risk", store, path)
            self.assertEqual(promoted["approved_risk_level"], "Medium Risk")
            self.assertEqual(len(store["pending_approvals"]), 0)
            self.assertEqual(len(store["approved_baseline"]), 1)

            candidate2 = dict(candidate)
            candidate2["id"] = "different-pending-id"
            semi_approve(candidate2, store, path)
            before_baseline = len(store["approved_baseline"])
            self.assertTrue(remove_pending(candidate2["id"], store, path))
            self.assertEqual(len(store["pending_approvals"]), 0)
            self.assertEqual(len(store["approved_baseline"]), before_baseline)

    def test_permanent_approval_requires_extracted_evidence(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            path = Path(temp) / "human_approval_store.json"
            store = load_review_store(path)
            with self.assertRaises(ValueError):
                approve_permanently({"id": "missing", "title": "Missing clause", "evidence": ""}, "Low Risk", store, path)



    def test_reject_remove_persists_without_changing_baseline_or_queue(self):
        from tempfile import TemporaryDirectory
        extracted, result = self._analyze_sample("high_risk_flawed_nda.pdf")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "human_approval_store.json"
            store = load_review_store(path)
            workflow = build_review_candidates(result, extracted["full_text"], "doc-hash", extracted["filename"], store, self.standards["clauses"])
            candidate = next(item for item in workflow["candidates"] if item.get("clause_id") == "D")
            self.assertTrue(reject_remove(candidate, store, path))
            reloaded = load_review_store(path)
            self.assertEqual(reloaded["approved_baseline"], [])
            self.assertEqual(reloaded["pending_approvals"], [])
            self.assertEqual(len(reloaded["dismissed_items"]), 1)
            again = build_review_candidates(result, extracted["full_text"], "doc-hash", extracted["filename"], reloaded, self.standards["clauses"])
            self.assertNotIn(candidate["id"], [item["id"] for item in again["candidates"]])

    def test_semi_approval_deduplicates_same_clause_across_documents(self):
        from tempfile import TemporaryDirectory
        extracted, result = self._analyze_sample("high_risk_flawed_nda.pdf")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "human_approval_store.json"
            store = load_review_store(path)
            workflow = build_review_candidates(result, extracted["full_text"], "doc-hash-1", extracted["filename"], store, self.standards["clauses"])
            candidate = next(item for item in workflow["candidates"] if item.get("clause_id") == "D")
            semi_approve(candidate, store, path)
            again = build_review_candidates(result, extracted["full_text"], "different-doc-hash", extracted["filename"], load_review_store(path), self.standards["clauses"])
            self.assertNotIn("D", [item.get("clause_id") for item in again["candidates"]])


    def test_bundled_evaluation_checks_pass(self):
        result = run_sample_evaluation(str(ROOT))
        self.assertEqual(result["checks_passed"], result["checks_total"], result)











if __name__ == "__main__":
    unittest.main()
