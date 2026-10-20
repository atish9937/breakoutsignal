import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd

import trend_pullback_backtest as backtest


class TrendPullbackBacktestTests(unittest.TestCase):
    def make_signal(self):
        return {
            "symbol": "TEST",
            "side": "CE",
            "date": "2025-01-02",
            "signal_time": "10:00:20",
            "signal_index": 0,
            "plan": {"stop": 98.0, "t1": 102.0, "t2": 104.0},
            "option": {"available": False},
        }

    def make_bars(self, rows):
        index = pd.date_range(
            "2025-01-02 10:00", periods=len(rows), freq="5min", tz=backtest.IST
        )
        return pd.DataFrame(
            rows, index=index, columns=["open", "high", "low", "close", "volume"]
        )

    def test_option_gate_uses_trend_strategy_atr_risk(self):
        setup = {
            "symbol": "TEST",
            "side": "CE",
            "ltp": 108.0,
            "plan": {"entry": 108.0, "risk_per_share": 2.0},
        }
        timestamp = pd.Timestamp("2025-01-02 10:00", tz=backtest.IST)
        snapshot = {
            "available": True,
            "wall": 110.0,
            "option_liquid": True,
            "strike": 110.0,
        }
        with patch.object(backtest.historical, "_option_snapshot", return_value=snapshot) as get:
            result = backtest._historical_option_check(setup, {}, timestamp)

        self.assertTrue(result["passes"])
        self.assertEqual(result["runway_r"], 1.0)
        self.assertEqual(get.call_args.args[4], 2.0)

    def test_insufficient_historical_oi_runway_fails_option_gate(self):
        setup = {
            "side": "CE",
            "ltp": 108.0,
            "plan": {"entry": 108.0, "risk_per_share": 2.0},
        }
        timestamp = pd.Timestamp("2025-01-02 10:00", tz=backtest.IST)
        snapshot = {"available": True, "wall": 109.0, "option_liquid": True}
        with patch.object(backtest.historical, "_option_snapshot", return_value=snapshot):
            result = backtest._historical_option_check(setup, {}, timestamp)

        self.assertFalse(result["passes"])
        self.assertEqual(result["runway_r"], 0.5)

    def test_trade_exits_half_at_t1_and_remainder_at_t2(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 100],
            [100, 102.5, 100.1, 102, 100],
            [102, 104.5, 101, 104, 100],
        ])

        trade = backtest._execute_trade(self.make_signal(), bars, 0, 0)

        self.assertEqual(trade["exit_reason"], "t2")
        self.assertEqual([fill["reason"] for fill in trade["exit_fills"]], ["t1", "t2"])
        self.assertAlmostEqual(trade["r_multiple"], 1.5)

    def test_stop_takes_priority_when_stop_and_t1_hit_in_same_bar(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 100],
            [100, 103, 97, 100, 100],
            [100, 100, 100, 100, 100],
        ])

        trade = backtest._execute_trade(self.make_signal(), bars, 0, 0)

        self.assertEqual(trade["exit_reason"], "stop")
        self.assertAlmostEqual(trade["r_multiple"], -1.0)

    def test_trade_is_skipped_when_next_open_invalidates_planned_stop(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 100],
            [97, 99, 96, 98, 100],
            [98, 100, 97, 99, 100],
        ])

        self.assertIsNone(backtest._execute_trade(self.make_signal(), bars, 0, 0))

    def test_scenario_does_not_reenter_same_side_after_first_eligible_signal(self):
        day = dt.date(2025, 1, 2)
        bars = self.make_bars([
            [100, 101, 99, 100, 100],
            [100, 101, 99, 100, 100],
            [100, 101, 99, 100, 100],
            [100, 101, 99, 100, 100],
            [100, 102.5, 100.1, 102, 100],
            [102, 104.5, 101, 104, 100],
        ])
        setup = {
            "symbol": "TEST",
            "side": "CE",
            "verdict": "SETUP_CE",
            "ltp": 100,
            "plan": {"entry": 100, "stop": 98, "t1": 102, "t2": 104,
                     "risk_per_share": 2},
        }
        data = {
            "TEST": {"intraday": bars, "options": {}},
            "NIFTY": {"intraday": bars},
        }
        snapshot = {
            "available": True,
            "wall": 102.0,
            "option_liquid": True,
            "strike": 100.0,
            "premium": 5.0,
            "oi": 1000.0,
            "iv": 20.0,
            "runway_r": 1.0,
            "passes": True,
        }
        with patch.object(backtest.strategy, "evaluate_setup", return_value=setup) as evaluate:
            with patch.object(backtest, "_historical_option_check", return_value=snapshot):
                with patch.object(backtest.historical, "_nifty_bias", return_value="UP"):
                    with patch.object(backtest.historical, "_option_pnl",
                                      return_value={"available": False}):
                        trades = backtest._scenario_trades(data, day, day, 0, 0)

        self.assertEqual(len(trades), 1)
        self.assertEqual(evaluate.call_count, 3)
        for call in evaluate.call_args_list:
            self.assertLessEqual(call.args[1].index[-1], call.args[2])


if __name__ == "__main__":
    unittest.main()
