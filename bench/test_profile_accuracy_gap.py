from pathlib import Path
import unittest

from profile_accuracy_gap import engine_profile, load_rows, paired_profile


def row(want, got, name_ok=False, exact_ok=False):
    return {
        "query": "q",
        "want": want,
        "got": got,
        "name_ok": name_ok,
        "exact_ok": exact_ok,
        "n_expected": len(want),
    }


class AccuracyGapProfileTest(unittest.TestCase):
    def test_separates_call_count_name_and_value_errors(self):
        want = [{"name": "a", "arguments": {"x": "right"}}]
        rows = [
            row(want, want, True, True),
            row(want, [], False, False),
            row(want, [{"name": "b", "arguments": {}}], False, False),
            row(want, [{"name": "a", "arguments": {"x": "wrong"}}], True, False),
        ]
        profile = engine_profile(rows)
        self.assertEqual(profile["exact"], 1)
        self.assertEqual(profile["error_classes"]["under_call"], 1)
        self.assertEqual(profile["error_classes"]["wrong_name"], 1)
        self.assertEqual(profile["error_classes"]["argument_values"], 1)

    def test_paired_net_uses_wins_as_well_as_losses(self):
        want = [{"name": "a", "arguments": {}}]
        exact = row(want, want, True, True)
        miss = row(want, [], False, False)
        candidate = [miss, exact, miss]
        reference = [exact, miss, exact]
        profile = paired_profile(candidate, reference)
        self.assertEqual(profile["reference_only"], 2)
        self.assertEqual(profile["candidate_only"], 1)
        self.assertEqual(profile["net_reference_advantage"], 1)
        self.assertEqual(profile["net_by_error_class"]["under_call"], 1)

    def test_published_final_profile_is_derived_from_raw_rows(self):
        results = Path(__file__).parent / "results"
        candidate = load_rows(
            results / "ours_961_final_20260822.json")
        reference = load_rows(results / "oracle_961_protocol_v2.json")

        candidate_profile = engine_profile(candidate)
        reference_profile = engine_profile(reference)
        paired = paired_profile(candidate, reference)
        self.assertEqual((candidate_profile["exact"], candidate_profile["names"]),
                         (669, 873))
        self.assertEqual((reference_profile["exact"], reference_profile["names"]),
                         (665, 943))
        self.assertEqual(paired["net_reference_advantage"], -4)


if __name__ == "__main__":
    unittest.main()
