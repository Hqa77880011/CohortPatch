import random
import unittest

from cohortpatch import core


class HistoricalTests(unittest.TestCase):
    def test_fixed_batch_policy_does_not_invent_unreleased_history_bits(self):
        case = next(core.generate(random.Random(17), 1))
        case['availability'] = [[]]
        case['forecasts'] = [[]]
        result = core.run_case(case, 'fixed_groups')
        self.assertEqual(result['minimum_class_size'], 12)
        self.assertEqual(result['final_classes'], 1)

    def test_roster_reservation_survives_crash_and_blocks_drift(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from cohortpatch.protocol import RosterLedger
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger.sqlite'
            ledger = RosterLedger(path, [list(range(6))], 3, 6, 3)
            ledger.reserve('one', [0, 1, 2], list(range(6)))
            ledger.close()
            ledger = RosterLedger(path, [list(range(6))], 3, 6, 3)
            with self.assertRaises(ValueError):
                ledger.freeze('one', [0, 1, 3])
            ledger.freeze('one', [0, 1, 2])
            values = {i: np.array([float(i)]) for i in range(6)}
            self.assertEqual(ledger.release('one', values).tolist(), [3.0])
            with self.assertRaises(ValueError):
                ledger.freeze('one', [0, 1, 2])
            self.assertEqual(ledger.release('one', {}).tolist(), [3.0])
            ledger.close()

    def test_policy_and_external_client_ids_are_validated(self):
        for field, value in (("k", 0), ("capacity", 0), ("threshold", 20)):
            case = next(core.generate(random.Random(17), 1))
            case[field] = value
            with self.assertRaises(ValueError):
                core.run_case(case, "cohortpatch")
        case = next(core.generate(random.Random(17), 1))
        case["availability"][0] = [case["clients"]]
        with self.assertRaisesRegex(ValueError, "enrolled"):
            core.run_case(case, "cohortpatch")
        case["availability"][0] = [0, 0]
        with self.assertRaisesRegex(ValueError, "unique"):
            core.run_case(case, "cohortpatch")

    def test_participation_tail_uses_usable_opportunities(self):
        case = dict(clients=6, k=3, threshold=3, capacity=3,
                    availability=[[0, 1, 2], list(range(6))], forecasts=[[], []])
        result = core.run_case(case, "cohortpatch")
        metrics = core.score(case, result)
        # Three clients included once in two usable rounds; three once in one.
        self.assertEqual(metrics["p10_inclusion_fraction"], 0.5)
        self.assertEqual(metrics["p10_inclusion_count"], 1.0)
        self.assertAlmostEqual(metrics["included_fraction"], 6 / 9)
        self.assertAlmostEqual(metrics["history_only_identity_accuracy"], 1 / 3)

    def test_two_sided_sizes(self):
        self.assertEqual(core.legal_size(8, 6, 3), [0, 3, 4, 5])
        self.assertEqual(core.legal_size(5, 4, 3), [0])
        self.assertEqual(core.legal_size(2, 2, 3), [0])

    def test_shared_warm_start_is_retained(self):
        case = next(core.generate(random.Random(17), 1))
        case['initial_classes'] = [list(range(6)), [6, 7, 8], [9, 10, 11]]
        case['availability'] = [[0, 1, 2, 6, 7, 8]]
        case['forecasts'] = [[]]
        result = core.run_case(case, 'cohortpatch')
        self.assertEqual(result['final_classes'], 4)
        self.assertEqual(result['minimum_class_size'], 3)
        self.assertEqual(result['violations'], 0)


if __name__ == '__main__':
    unittest.main()
