import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd

import breakout_scanner as core
import trend_pullback as strategy


class TrendPullbackTests(unittest.TestCase):
    def make_candles(self):
        prior_index = pd.date_range(
            "2026-10-16 10:00", periods=55, freq="5min", tz=core.IST
        )
        today_index = pd.date_range(
            "2026-10-19 10:00", periods=5, freq="5min", tz=core.IST
        )
        index = prior_index.append(today_index)
        rows = [[100, 101, 99, 100, 100] for _ in prior_index]
        rows.extend([
            [110, 111, 109, 110, 100],
            [107, 110, 105, 107, 100],
            [105, 108, 104, 105, 100],
            [106, 107, 105, 106, 100],
            [107, 109, 106, 108, 120],
        ])
        return pd.DataFrame(
            rows, index=index, columns=["open", "high", "low", "close", "volume"]
        )

    def evaluate(self, bias="UP"):
        candles = self.make_candles()

        def fake_ema(values, span):
            level = {9: 107.5, 21: 105.0, 50: 100.0}[span]
            return pd.Series(level, index=values.index)

        now = dt.datetime(2026, 10, 19, 10, 25, tzinfo=core.IST)
        with patch.object(core, "ema", side_effect=fake_ema), \
                patch.object(core, "atr", return_value=2.0), \
                patch.object(core, "vwap", return_value=103.0), \
                patch.object(core, "rsi", return_value=60.0), \
                patch.object(core, "macd_hist", return_value=0.2), \
                patch.object(core, "adx", return_value=25.0), \
                patch.object(core, "momentum_divergence", return_value=None):
            result = strategy.evaluate_setup(
                "TEST", candles, now, {"bias": bias}
            )
        return result

    def test_bullish_pullback_passes_all_underlying_gates(self):
        result = self.evaluate()

        self.assertEqual(result["side"], "CE")
        self.assertEqual(result["verdict"], "SETUP_CE")
        self.assertTrue(all(check["passed"] for check in result["mechanical_checks"]))
        self.assertEqual(result["plan"]["stop"], 106.0)
        self.assertEqual(result["plan"]["t2"], 112.0)

    def test_opposing_nifty_bias_prevents_setup(self):
        result = self.evaluate(bias="DOWN")

        self.assertEqual(result["verdict"], "WATCH_CE")
        self.assertIn("Nifty alignment", result["fails"])

    def test_bearish_pullback_selects_put_and_checks_downtrend(self):
        candles = self.make_candles()
        candles.loc[candles.index[-5]:, ["open", "high", "low", "close", "volume"]] = [
            [90, 91, 89, 90, 100],
            [93, 95, 90, 93, 100],
            [95, 96, 92, 95, 100],
            [94, 95, 93, 94, 100],
            [92, 94, 91, 92, 120],
        ]

        def fake_ema(values, span):
            level = {9: 92.5, 21: 95.0, 50: 100.0}[span]
            return pd.Series(level, index=values.index)

        now = dt.datetime(2026, 10, 19, 10, 25, tzinfo=core.IST)
        with patch.object(core, "ema", side_effect=fake_ema), \
                patch.object(core, "atr", return_value=2.0), \
                patch.object(core, "vwap", return_value=97.0), \
                patch.object(core, "rsi", return_value=40.0), \
                patch.object(core, "macd_hist", return_value=-0.2), \
                patch.object(core, "adx", return_value=25.0), \
                patch.object(core, "momentum_divergence", return_value=None):
            result = strategy.evaluate_setup("TEST", candles, now, {"bias": "DOWN"})

        self.assertEqual(result["side"], "PE")
        self.assertEqual(result["verdict"], "SETUP_PE")
        self.assertTrue(all(check["passed"] for check in result["mechanical_checks"]))

    def test_outside_entry_window_does_not_create_setup(self):
        late_now = dt.datetime(2026, 10, 19, 12, 35, tzinfo=core.IST)
        candles = self.make_candles()

        def fake_ema(values, span):
            return pd.Series({9: 107.5, 21: 105.0, 50: 100.0}[span], index=values.index)

        with patch.object(core, "ema", side_effect=fake_ema), \
                patch.object(core, "atr", return_value=2.0), \
                patch.object(core, "vwap", return_value=103.0), \
                patch.object(core, "rsi", return_value=60.0), \
                patch.object(core, "macd_hist", return_value=0.2):
            result = strategy.evaluate_setup("TEST", candles, late_now, {"bias": "UP"})

        self.assertEqual(result["verdict"], "WATCH_CE")
        self.assertIn("Entry time", result["fails"])

    def test_empty_candle_response_returns_a_clear_skip(self):
        now = dt.datetime(2026, 10, 19, 10, 25, tzinfo=core.IST)
        result = strategy.evaluate_setup(
            "TEST",
            pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
            now,
            {"bias": "FLAT"},
        )

        self.assertEqual(result["verdict"], "SKIP")
        self.assertIn("no timestamped candle data", result["fails"])

    def test_option_gates_promote_setup_to_recommendation(self):
        result = self.evaluate()
        option_info = {
            "strike": 110.0,
            "wall": 112.0,
            "opposite_wall": 100.0,
            "opt": {
                "last_price": 10.0,
                "top_bid_price": 9.8,
                "top_ask_price": 10.2,
                "oi": 1000,
                "greeks": {"delta": 0.5},
            },
        }
        with patch.object(core, "pick_option", return_value=option_info), \
                patch.object(core, "lot_and_id", return_value=(50, 12345)):
            result = strategy.attach_option_check(
                result, None, "2026-10-29", {}, day=dt.date(2026, 10, 19)
            )

        self.assertEqual(result["verdict"], "BUY_CE")
        self.assertEqual(result["option"]["security_id"], 12345)
        self.assertEqual(result["plan"]["runway_r"], 2.0)

    def test_insufficient_oi_runway_blocks_recommendation(self):
        result = self.evaluate()
        option_info = {
            "strike": 110.0,
            "wall": 109.0,
            "opposite_wall": 100.0,
            "opt": {
                "last_price": 10.0,
                "top_bid_price": 9.8,
                "top_ask_price": 10.2,
                "oi": 1000,
                "greeks": {"delta": 0.5},
            },
        }
        with patch.object(core, "pick_option", return_value=option_info), \
                patch.object(core, "lot_and_id", return_value=(50, 12345)):
            result = strategy.attach_option_check(
                result, None, "2026-10-29", {}, day=dt.date(2026, 10, 19)
            )

        self.assertEqual(result["verdict"], "WATCH_CE")
        self.assertIn("OI-wall runway", result["fails"])

    def test_option_spread_blocks_recommendation(self):
        result = self.evaluate()
        option_info = {
            "strike": 110.0,
            "wall": 112.0,
            "opposite_wall": 100.0,
            "opt": {
                "last_price": 10.0,
                "top_bid_price": 8.0,
                "top_ask_price": 10.5,
                "oi": 1000,
                "greeks": {"delta": 0.5},
            },
        }
        with patch.object(core, "pick_option", return_value=option_info), \
                patch.object(core, "lot_and_id", return_value=(50, 12345)):
            result = strategy.attach_option_check(
                result, None, "2026-10-29", {}, day=dt.date(2026, 10, 19)
            )

        self.assertEqual(result["verdict"], "WATCH_CE")
        self.assertIn("Option liquidity", result["fails"])

    def _wrap_attach(self, opt_overrides, *, day=dt.date(2026, 10, 19),
                     expiry="2026-10-29"):
        result = self.evaluate()
        base = {
            "last_price": 10.0,
            "top_bid_price": 9.9,
            "top_ask_price": 10.1,
            "oi": 2000,
            "greeks": {"delta": 0.5},
        }
        base.update(opt_overrides or {})
        option_info = {
            "strike": 110.0, "wall": 112.0, "opposite_wall": 100.0, "opt": base,
        }
        with patch.object(core, "pick_option", return_value=option_info), \
                patch.object(core, "lot_and_id", return_value=(50, 12345)):
            return strategy.attach_option_check(result, None, expiry, {}, day=day)

    def test_zero_dte_contract_fails_liquidity_gate(self):
        result = self._wrap_attach({}, expiry="2026-10-19")
        self.assertEqual(result["verdict"], "WATCH_CE")
        liquidity = next(c for c in result["mechanical_checks"]
                         if c["gate"] == "Option liquidity")
        self.assertFalse(liquidity["passed"])
        self.assertIn("DTE 0d", liquidity["current"])

    def test_zero_bid_fails_liquidity_gate(self):
        result = self._wrap_attach({"top_bid_price": 0.0})
        liquidity = next(c for c in result["mechanical_checks"]
                         if c["gate"] == "Option liquidity")
        self.assertFalse(liquidity["passed"])
        self.assertEqual(result["verdict"], "WATCH_CE")

    def test_low_delta_fails_liquidity_gate(self):
        result = self._wrap_attach({"greeks": {"delta": 0.15}})
        liquidity = next(c for c in result["mechanical_checks"]
                         if c["gate"] == "Option liquidity")
        self.assertFalse(liquidity["passed"])
        self.assertEqual(result["verdict"], "WATCH_CE")

    def test_penny_premium_fails_liquidity_gate(self):
        result = self._wrap_attach({"last_price": 0.5})
        liquidity = next(c for c in result["mechanical_checks"]
                         if c["gate"] == "Option liquidity")
        self.assertFalse(liquidity["passed"])
        self.assertEqual(result["verdict"], "WATCH_CE")

    def test_low_oi_fails_liquidity_gate(self):
        result = self._wrap_attach({"oi": 500})
        liquidity = next(c for c in result["mechanical_checks"]
                         if c["gate"] == "Option liquidity")
        self.assertFalse(liquidity["passed"])
        self.assertEqual(result["verdict"], "WATCH_CE")

    def _evaluate_with_overrides(self, *, adx_value=25.0, divergence=None,
                                 daily_ctx=None, breadth=None, vix=None,
                                 bias="UP"):
        candles = self.make_candles()

        def fake_ema(values, span):
            level = {9: 107.5, 21: 105.0, 50: 100.0}[span]
            return pd.Series(level, index=values.index)

        now = dt.datetime(2026, 10, 19, 10, 25, tzinfo=core.IST)
        with patch.object(core, "ema", side_effect=fake_ema), \
                patch.object(core, "atr", return_value=2.0), \
                patch.object(core, "vwap", return_value=103.0), \
                patch.object(core, "rsi", return_value=60.0), \
                patch.object(core, "macd_hist", return_value=0.2), \
                patch.object(core, "adx", return_value=adx_value), \
                patch.object(core, "momentum_divergence", return_value=divergence):
            return strategy.evaluate_setup(
                "TEST", candles, now, {"bias": bias},
                daily_ctx=daily_ctx, breadth=breadth, vix=vix,
            )

    def test_low_adx_vetoes_setup(self):
        result = self._evaluate_with_overrides(adx_value=12.0)
        adx_check = next(c for c in result["mechanical_checks"] if c["gate"] == "ADX trend strength")
        self.assertFalse(adx_check["passed"])
        self.assertEqual(result["verdict"], "WATCH_CE")

    def test_bearish_divergence_vetoes_ce(self):
        result = self._evaluate_with_overrides(
            divergence="bearish RSI divergence: price HH but RSI LH"
        )
        div_check = next(c for c in result["mechanical_checks"] if c["gate"] == "Momentum divergence")
        self.assertFalse(div_check["passed"])
        self.assertEqual(result["verdict"], "WATCH_CE")

    def test_breadth_below_threshold_vetoes_ce(self):
        result = self._evaluate_with_overrides(
            breadth={"pct_above_prev_close": 40.0}
        )
        breadth_check = next(c for c in result["mechanical_checks"] if c["gate"] == "Market breadth")
        self.assertFalse(breadth_check["passed"])
        self.assertEqual(result["verdict"], "WATCH_CE")

    def test_vix_spike_vetoes_entry(self):
        result = self._evaluate_with_overrides(
            vix={"ltp": 28.0, "ema20": 15.0}  # 1.87x spike
        )
        vix_check = next(c for c in result["mechanical_checks"] if c["gate"] == "India VIX regime")
        self.assertFalse(vix_check["passed"])
        self.assertEqual(result["verdict"], "WATCH_CE")

    def test_daily_trend_wrong_direction_vetoes_ce(self):
        # Daily EMA20 slope negative -> CE should be vetoed
        slope_series = pd.Series([105, 104, 103, 102, 101, 100], dtype=float)
        result = self._evaluate_with_overrides(
            daily_ctx={"ema20_series": slope_series}
        )
        daily_check = next(c for c in result["mechanical_checks"] if c["gate"] == "Daily trend")
        self.assertFalse(daily_check["passed"])
        self.assertEqual(result["verdict"], "WATCH_CE")

    def test_missing_breadth_or_vix_does_not_block_setup(self):
        """Graceful degradation: when external context is unavailable the gate becomes a no-op."""
        result = self._evaluate_with_overrides(breadth=None, vix=None, daily_ctx=None)
        self.assertEqual(result["verdict"], "SETUP_CE")


if __name__ == "__main__":
    unittest.main()
