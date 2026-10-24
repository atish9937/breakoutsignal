import unittest

import validation_report as vr


class ValidationReportTests(unittest.TestCase):
    def row(self, status):
        return {"reference_entry": 100.0, "stop_price": 90.0, "target_1": 115.0,
                "target_2": 125.0, "status": status}

    def test_r_multiples_follow_half_t1_half_runner_plan(self):
        self.assertEqual(vr._r_multiple(self.row("stop_before_t1")), -1.0)
        self.assertAlmostEqual(vr._r_multiple(self.row("t1_then_breakeven")), 0.75)
        self.assertAlmostEqual(vr._r_multiple(self.row("t2_hit")), 2.0)
        self.assertIsNone(vr._r_multiple(self.row("tracking")))

    def test_stats_profit_factor_and_drawdown(self):
        stats = vr._stats([2.0, -1.0, -1.0, 1.0])
        self.assertEqual(stats["profit_factor"], 1.5)
        self.assertEqual(stats["max_drawdown"], 2.0)
        self.assertEqual(stats["win_rate_pct"], 50.0)

    def test_empty_ledger_builds_without_error(self):
        import sqlite3
        import trade_performance
        with trade_performance.connect(vr.Path(":memory:")) as connection:
            result = vr.build(connection)
        self.assertFalse(result["strategies"]["breakout"]["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
