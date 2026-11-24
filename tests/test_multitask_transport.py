"""Unit tests for the multitask sidecar's received-candidate echo."""

from __future__ import annotations

import unittest

from biod.multitask.serve import _received_transport


class ReceivedTransportTest(unittest.TestCase):
    def test_description_cards_are_counted(self) -> None:
        payload = {
            "candidates": [
                {"id": "a", "name": "A", "description": "abc"},
                {"id": "b", "name": "B", "description": "de"},
            ],
            "catalog": {"fingerprint": "f" * 64, "total": 5, "shortlist": 2, "truncated": True},
        }
        got = _received_transport(payload)
        self.assertEqual(got["mode"], "descriptions")
        self.assertEqual(got["candidateCount"], 2)
        self.assertEqual(got["descriptionChars"], 5)
        self.assertTrue(got["truncated"])
        self.assertEqual(got["catalogFingerprint"], "f" * 64)
        self.assertEqual(got["catalogTotal"], 5)
        self.assertEqual(got["shortlist"], 2)

    def test_id_only_payload_is_marked_ids(self) -> None:
        got = _received_transport({"candidates": ["a", "b"]})
        self.assertEqual(got["mode"], "ids")
        self.assertEqual(got["descriptionChars"], 0)
        self.assertEqual(got["candidateCount"], 2)

    def test_missing_fields_default_to_the_received_count(self) -> None:
        got = _received_transport({"candidates": [{"id": "a", "description": "x"}]})
        self.assertEqual(got["catalogTotal"], 1)
        self.assertEqual(got["shortlist"], 1)
        self.assertFalse(got["truncated"])


if __name__ == "__main__":
    unittest.main()
