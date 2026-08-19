import unittest

from device_runner import stratified_sample, wilson_interval
from mobile_actions import build_cases, common_bm25_tools, encode_field, score


class MobileActionsProtocolTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "tools": [
                    {"function": {"name": "z", "parameters": {"type": "OBJECT"}}},
                    {"function": {"name": "a", "parameters": {"type": "OBJECT"}}},
                ],
                "messages": [
                    {"role": "developer", "content": "date: now\nlocale: en"},
                    {"role": "user", "content": "run it"},
                ],
            },
            {
                "tools": [
                    {"function": {"name": "a", "parameters": {"type": "OBJECT"}}},
                    {"function": {"name": "z", "parameters": {"type": "OBJECT"}}},
                ],
                "messages": [
                    {"role": "developer", "content": "date: later"},
                    {"role": "user", "content": "run it again"},
                ],
            },
        ]

    def test_tool_order_modes(self):
        dataset = build_cases(self.records, "dataset")
        canonical = build_cases(self.records, "canonical")
        fixed = build_cases(self.records, "fixed-first")
        self.assertEqual([[tool["name"] for tool in case[2]] for case in dataset],
                         [["z", "a"], ["a", "z"]])
        self.assertEqual([[tool["name"] for tool in case[2]] for case in canonical],
                         [["a", "z"], ["a", "z"]])
        self.assertEqual([[tool["name"] for tool in case[2]] for case in fixed],
                         [["z", "a"], ["z", "a"]])

    def test_strict_and_historical_normalized_metrics(self):
        want = [{"name": "send", "arguments": {"to": "Ada@Example.com"}}]
        got = [{"name": "send", "arguments": {"to": "ada@example.com"}}]
        self.assertEqual(score(got, want, "strict"), (True, False))
        self.assertEqual(score(got, want, "normalized"), (True, True))

    def test_line_protocol_preserves_newlines(self):
        self.assertEqual(encode_field("one\ntwo\tthree"), "one\x1etwo three")

    def test_common_retrieval_is_stable_and_bounded(self):
        tools = [
            {"name": "mail", "description": "send an email"},
            {"name": "map", "description": "show a location"},
            {"name": "timer", "description": "start a countdown"},
        ]
        selected = common_bm25_tools("email this report", tools)
        self.assertEqual([tool["name"] for tool in selected], ["mail", "map"])

    def test_device_sample_is_proportional_and_reproducible(self):
        def record(call_count):
            calls = [{"function": {"name": f"call_{i}", "arguments": {}}}
                     for i in range(call_count)]
            return {"messages": [{"role": "assistant", "tool_calls": calls}]}

        population = list(enumerate(
            [record(1) for _ in range(640)] +
            [record(2) for _ in range(320)] +
            [record(3)],
        ))
        first, allocation = stratified_sample(population, 12, 20260819)
        second, _ = stratified_sample(population, 12, 20260819)
        self.assertEqual(allocation, {1: 8, 2: 4, 3: 0})
        self.assertEqual([index for index, _ in first], [index for index, _ in second])

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(6, 12)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


if __name__ == "__main__":
    unittest.main()
