import unittest
from learned_clause_model import predict_clause, model_status
from analyzer import calculate_risk_score


class LearnedModelTests(unittest.TestCase):
    def test_local_classifier_recognises_paraphrased_exclusion(self):
        text = ("Information already known to the recipient before disclosure or independently "
                "developed without using the disclosed material is excluded from the duty.")
        result = predict_clause(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["clause_id"], "C")

    def test_local_classifier_recognises_return_deletion_paraphrase(self):
        text = ("At the end of this agreement the receiving party shall erase all electronic copies "
                "and deliver back every document within thirty days.")
        result = predict_clause(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["clause_id"], "E")

    def test_local_classifier_abstains_on_unrelated_text(self):
        text = "The supplier shall deliver the equipment to the customer by the end of next month."
        self.assertIsNone(predict_clause(text))

    def test_model_is_bundled_and_loadable(self):
        self.assertTrue(model_status()["available"])

    def test_primary_score_uses_final_clause_risk_labels_not_fuzzy_score(self):
        clauses = ([{"clause_id": f"H{i}", "risk_level": "High Risk", "status": "Needs Review", "fuzzy_risk_score": 10} for i in range(5)] +
                   [{"clause_id": f"M{i}", "risk_level": "Medium Risk", "status": "Needs Review", "fuzzy_risk_score": 10} for i in range(4)] +
                   [{"clause_id": "L", "risk_level": "Low Risk", "status": "Present", "fuzzy_risk_score": 10}])
        standards = [{"id": c["clause_id"], "weight": 1.0} for c in clauses]
        result = calculate_risk_score(clauses, standards)
        self.assertEqual(result["overall_score"], 66)
        self.assertEqual(result["overall_category"], "High Risk")
        self.assertEqual(result["fuzzy_overall_score"], 10.0)


if __name__ == "__main__":
    unittest.main()
