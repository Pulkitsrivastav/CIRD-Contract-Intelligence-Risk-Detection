"""Focused regression tests for contextual offline clause screening.

These are controlled phrase/section tests, not a statistically representative
legal benchmark. They protect against common lexical false positives/negatives.
"""
from __future__ import annotations

import unittest

from analyzer import analyze_with_keywords, load_standard_clauses


class OfflineContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.standards = load_standard_clauses("data/nda_standards.json")

    def clause(self, text: str, clause_id: str) -> dict:
        result = analyze_with_keywords(text, self.standards)
        self.assertTrue(result.get("success"), result.get("error"))
        return next(item for item in result["clauses"] if item["clause_id"] == clause_id)

    def test_paraphrased_return_and_destroy_across_multiple_sentences(self):
        text = ("--- Page 1 ---\nReturn and Deletion\n"
                "Upon expiry, the recipient shall hand back all protected materials. "
                "Electronic copies must be permanently erased within 30 days.")
        result = self.clause(text, "E")
        self.assertEqual(result["status"], "Present")
        self.assertIn("return_or_destroy_action", [x["concept"] for x in result["matched_concepts"]])
        self.assertTrue(result["evidence_verified"])

    def test_combined_sentences_identify_defined_limited_use(self):
        text = ("--- Page 1 ---\nUse of Information\n"
                "The parties are considering a possible commercial partnership. "
                "Information received may only be used to evaluate that proposed relationship. "
                "It shall not be used for any unrelated purpose.")
        result = self.clause(text, "B")
        self.assertEqual(result["status"], "Present")
        concepts = {x["concept"] for x in result["matched_concepts"]}
        self.assertIn("defined_business_purpose", concepts)
        self.assertIn("limited_use", concepts)

    def test_single_generic_terms_do_not_prove_clause_presence(self):
        text = ("--- Page 1 ---\nThe company has security staff, financial plans, customer lists, "
                "and a five-year business strategy. The parties discussed liability and termination dates.")
        for clause_id in ("A", "D", "G", "H", "J"):
            result = self.clause(text, clause_id)
            self.assertEqual(result["status"], "Missing / Not Detected", (clause_id, result))

    def test_negated_perpetual_duration_is_not_flagged_as_perpetual(self):
        text = ("--- Page 1 ---\nConfidentiality Period\n"
                "The confidentiality obligations shall not continue in perpetuity and will expire "
                "three years after termination.")
        result = self.clause(text, "D")
        self.assertNotIn("perpetual confidentiality", result["matched_red_flags"])
        self.assertNotEqual(result["risk_level"], "High Risk")

    def test_unlimited_liability_paraphrase_is_detected(self):
        text = ("--- Page 1 ---\nLiability\n"
                "Each party shall be liable without any cap for all losses arising from any breach of this agreement.")
        result = self.clause(text, "H")
        self.assertIn("unlimited liability", result["matched_red_flags"])
        self.assertEqual(result["risk_level"], "High Risk")

    def test_negated_security_red_flag_does_not_trigger(self):
        text = ("--- Page 1 ---\nSecurity\n"
                "The recipient shall maintain reasonable care and appropriate technical safeguards. "
                "There is no absence of security obligations under this agreement.")
        result = self.clause(text, "G")
        self.assertNotIn("no security obligation", result["matched_red_flags"])


    def test_unrestricted_third_party_disclosure_is_high_risk(self):
        text = ("--- Page 1 ---\n9. Disclosure to Third Parties\n"
                "The recipient may disclose Confidential Information to third parties where it considers "
                "such disclosure useful for the proposed commercial relationship.")
        result = self.clause(text, "F")
        self.assertIn("unrestricted third-party disclosure", result["matched_red_flags"])
        self.assertEqual(result["risk_level"], "High Risk")
        self.assertEqual(result["status"], "Needs Review")

    def test_broad_uncapped_liability_wording_is_high_risk(self):
        text = ("--- Page 1 ---\n8. Liability and Damages\n"
                "Each recipient shall be responsible for all losses, claims, costs, and damages arising "
                "from any breach of this Agreement. Neither party shall be liable for indirect damages.")
        result = self.clause(text, "H")
        self.assertIn("broad liability without an express cap", result["matched_red_flags"])
        self.assertEqual(result["risk_level"], "High Risk")

    def test_evidence_is_a_source_excerpt_not_an_invented_summary(self):
        text = "--- Page 1 ---\nPermitted Use\nThe information may only be used to evaluate the proposed transaction."
        result = self.clause(text, "B")
        self.assertTrue(result["evidence_verified"])
        self.assertIn("may only be used", result["extracted_evidence"])


if __name__ == "__main__":
    unittest.main()
