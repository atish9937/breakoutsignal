import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import swing_option_backtest as backtest
import swing_option_strategy as strategy


IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


class SwingOptionStrategyTests(unittest.TestCase):
    def make_daily(self):
        index = pd.date_range("2026-01-01", periods=70, freq="B", tz=IST)
        close = [100 + value * 0.5 for value in range(70)]
        close[-3:] = [150, 154, 160]
        high = [value + 1 for value in close]
        low = [value - 1 for value in close]
        low[-2] = 150.5
        high[-2] = 155
        volume = [100.0] * 70
        volume[-1] = 200
        return pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )

    def test_generates_daily_pullback_and_breakout_candidates_with_confirmations(self):
        daily = self.make_daily()
        nifty = daily.copy()
        nifty["close"] = [100 + value for value in range(len(nifty))]

        def fake_ema(values, span):
            level = 155 if span == 20 else 140
            output = pd.Series(level, index=values.index, dtype=float)
            if span == 20:
                output.iloc[-6:] = [154, 154.2, 154.4, 154.6, 154.8, 155]
            return output

        with patch.object(strategy.core, "ema", side_effect=fake_ema), \
                patch.object(strategy.core, "atr", return_value=5.0), \
                patch.object(strategy.core, "rsi", return_value=60.0):
            candidates = strategy.evaluate_setups(
                "TEST", daily, nifty, daily.index[-1].date()
            )

        self.assertEqual(
            {candidate["setup"] for candidate in candidates},
            {"Daily pullback and reversal", "Daily range breakout"},
        )
        self.assertTrue(all(candidate["side"] == "CE" for candidate in candidates))
        self.assertTrue(all(candidate["holding"]["weekend_carry"] is False
                            for candidate in candidates))

    def test_option_contract_choice_prioritizes_spread_after_hard_liquidity_gates(self):
        chain = {
            "oc": {
                "100": {"ce": {"last_price": 12, "top_bid_price": 11.95,
                               "top_ask_price": 12.05, "oi": 2000, "volume": 100,
                               "greeks": {"delta": 0.6}}},
                "105": {"ce": {"last_price": 10, "top_bid_price": 9.9,
                               "top_ask_price": 10.1, "oi": 1500, "volume": 80,
                               "greeks": {"delta": 0.5}}},
            }
        }
        master = pd.DataFrame()
        with patch.object(strategy.core, "lot_and_id", return_value=(10, 123)):
            options = strategy._option_candidates(
                chain, "CE", 105, "TEST", "2026-02-26", master
            )
        self.assertEqual([option["selection"] for option in options], ["ATM", "slightly_ITM"])
        self.assertTrue(all(option["passes"] for option in options))
        selected = min(options, key=lambda item: item["spread_pct"])
        self.assertEqual(selected["selection"], "slightly_ITM")

    def test_daily_report_saves_ranked_candidate_and_selects_tighter_live_spread(self):
        candidate = {
            "symbol": "TEST",
            "side": "CE",
            "setup": "Daily pullback and reversal",
            "score": 7,
            "underlying": {"close": 100},
        }
        daily = self.make_daily()
        dh = type("DhanStub", (), {
            "daily": lambda self, *args, **kwargs: daily,
            "expiries": lambda self, sid: ["2026-10-27", "2026-11-05"],
            "option_chain": lambda self, sid, expiry: {"oc": {}},
        })()
        option_candidates = [
            {"selection": "ATM", "spread_pct": 3.0, "passes": True,
             "open_interest": 100, "volume": 100, "ask": 11},
            {"selection": "slightly_ITM", "spread_pct": 1.0, "passes": True,
             "open_interest": 80, "volume": 90, "ask": 12},
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                strategy, "evaluate_setups", return_value=[candidate]
            ), patch.object(
                strategy, "_option_candidates", return_value=option_candidates
            ):
                report = strategy.scan_daily_candidates(
                    dh, pd.DataFrame(), {"TEST": 123},
                    dt.date(2026, 10, 22), Path(directory),
                )
            saved = Path(report["output_path"])
            self.assertTrue(saved.exists())
            self.assertEqual(report["candidates"][0]["expiry"], "2026-11-05")
            self.assertEqual(
                report["candidates"][0]["selected_option"]["selection"],
                "slightly_ITM",
            )

    def test_expiry_helper_rejects_expiry_too_close_for_multi_session_hold(self):
        as_of = dt.date(2026, 10, 22)
        self.assertEqual(
            strategy._expiry_at_least(
                ["2026-10-27", "2026-11-05", "invalid"], as_of
            ),
            "2026-11-05",
        )

    def test_option_premium_exit_prioritizes_stop_on_ambiguous_candle(self):
        timestamps = pd.date_range("2026-01-05 09:15", periods=2, freq="5min", tz=IST)
        signal_day = dt.date(2026, 1, 2)
        underlying = pd.DataFrame(
            [[100, 101, 99, 101, 100], [101, 102, 100, 101, 100]],
            index=timestamps,
            columns=["open", "high", "low", "close", "volume"],
        )
        options = {
            "CALL_ATM": pd.DataFrame(
                [[10, 11, 9, 10, 100, 1000, 20, 100, 100],
                 [10, 17, 6, 11, 100, 1000, 20, 100, 100]],
                index=timestamps,
                columns=["open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot"],
            )
        }
        candidate = {
            "symbol": "TEST",
            "side": "CE",
            "setup": "Daily pullback and reversal",
            "signal_date": signal_day.isoformat(),
            "underlying": {"entry_trigger": 100, "stop": 95, "close": 99},
        }
        contract = {"option_type": "CALL", "strike": 100}
        contract["selection"] = "ATM"

        trade = backtest._execute_candidate(
            candidate, underlying, options, contract, "option_premium", 0, 0
        )

        self.assertEqual(trade["exit_reason"], "premium_stop")
        self.assertEqual(trade["exit_time"], timestamps[1].isoformat())
        self.assertLess(trade["return_pct"], 0)

    def test_underlying_exit_uses_stop_when_stop_and_target_share_a_bar(self):
        timestamps = pd.date_range("2026-01-05 09:15", periods=3, freq="5min", tz=IST)
        underlying = pd.DataFrame(
            [[100, 101, 99, 101, 100], [101, 120, 90, 105, 100],
             [105, 106, 104, 105, 100]],
            index=timestamps,
            columns=["open", "high", "low", "close", "volume"],
        )
        options = {
            "CALL_ATM": pd.DataFrame(
                [[10, 11, 9, 10, 100, 1000, 20, 100, 100],
                 [10, 20, 5, 8, 100, 1000, 20, 100, 100],
                 [11, 12, 10, 11, 100, 1000, 20, 100, 100]],
                index=timestamps,
                columns=["open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot"],
            )
        }
        candidate = {
            "symbol": "TEST",
            "side": "CE",
            "setup": "Daily range breakout",
            "signal_date": "2026-01-02",
            "underlying": {"entry_trigger": 100, "stop": 95, "close": 99},
        }
        trade = backtest._execute_candidate(
            candidate, underlying, options,
            {"option_type": "CALL", "strike": 100, "selection": "ATM"},
            "underlying_atr", 0, 0,
        )
        self.assertEqual(trade["exit_reason"], "underlying_stop")
        self.assertLess(trade["return_pct"], 0)

    def test_three_session_holding_window_uses_trading_dates_not_calendar_days(self):
        index = pd.DatetimeIndex([
            "2026-01-02 09:15",
            "2026-01-05 09:15",
            "2026-01-06 09:15",
            "2026-01-07 09:15",
            "2026-01-08 09:15",
        ], tz=IST)
        frame = pd.DataFrame(
            [[1, 1, 1, 1, 1]] * len(index),
            index=index,
            columns=["open", "high", "low", "close", "volume"],
        )
        future = backtest._future_bars(
            frame, pd.Timestamp("2026-01-05 09:15", tz=IST)
        )
        self.assertEqual(list(dict.fromkeys(future.index.date)), [
            dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 7),
        ])

    def test_holding_window_stops_at_friday_close(self):
        index = pd.DatetimeIndex([
            "2026-01-08 09:15",
            "2026-01-09 09:15",
            "2026-01-12 09:15",
            "2026-01-13 09:15",
        ], tz=IST)
        frame = pd.DataFrame(
            [[1, 1, 1, 1, 1]] * len(index),
            index=index,
            columns=["open", "high", "low", "close", "volume"],
        )
        future = backtest._future_bars(
            frame, pd.Timestamp("2026-01-08 09:15", tz=IST)
        )
        self.assertEqual(list(dict.fromkeys(future.index.date)), [
            dt.date(2026, 1, 8), dt.date(2026, 1, 9),
        ])


if __name__ == "__main__":
    unittest.main()
