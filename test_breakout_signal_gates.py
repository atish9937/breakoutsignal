import unittest

import pandas as pd

import breakout_scanner


class BreakoutSignalGateTests(unittest.TestCase):
    def test_live_price_must_remain_beyond_call_trigger(self):
        self.assertTrue(breakout_scanner._live_price_beyond_trigger("CE", 101, 100))
        self.assertFalse(breakout_scanner._live_price_beyond_trigger("CE", 100, 100))
        self.assertFalse(breakout_scanner._live_price_beyond_trigger("CE", 99, 100))

    def test_live_price_must_remain_beyond_put_trigger(self):
        self.assertTrue(breakout_scanner._live_price_beyond_trigger("PE", 99, 100))
        self.assertFalse(breakout_scanner._live_price_beyond_trigger("PE", 100, 100))
        self.assertFalse(breakout_scanner._live_price_beyond_trigger("PE", 101, 100))


class ADXHelperTests(unittest.TestCase):
    def test_strongly_trending_market_registers_high_adx(self):
        closes = pd.Series([100 + i * 0.3 for i in range(60)], dtype=float)
        candles = pd.DataFrame({
            "open": closes, "high": closes + 1, "low": closes - 1, "close": closes
        })
        value = breakout_scanner.adx(candles)
        self.assertIsNotNone(value)
        self.assertGreater(value, 50)  # trending market has ADX well above 20

    def test_flat_market_registers_zero_adx(self):
        closes = pd.Series([100] * 60, dtype=float)
        candles = pd.DataFrame({
            "open": closes, "high": closes + 0.1, "low": closes - 0.1, "close": closes
        })
        value = breakout_scanner.adx(candles)
        self.assertIsNotNone(value)
        self.assertLess(value, 5)

    def test_short_history_returns_none(self):
        closes = pd.Series([100, 101, 102], dtype=float)
        candles = pd.DataFrame({
            "open": closes, "high": closes + 1, "low": closes - 1, "close": closes
        })
        self.assertIsNone(breakout_scanner.adx(candles, n=14))


class DivergenceHelperTests(unittest.TestCase):
    def test_bearish_divergence_is_detected_for_call_side(self):
        closes = pd.Series(
            [100] * 20
            + [80, 85, 90, 95, 100, 105, 110, 108, 106, 104, 102, 104, 106, 108, 110, 109, 110.3],
            dtype=float,
        )
        candles = pd.DataFrame({
            "open": closes, "high": closes + 0.5, "low": closes - 0.5, "close": closes
        })
        result = breakout_scanner.momentum_divergence(candles, "CE", lookback=14)
        self.assertIsNotNone(result)
        self.assertIn("bearish RSI divergence", result)

    def test_clean_uptrend_shows_no_divergence(self):
        closes = pd.Series([100 + i * 0.4 for i in range(50)], dtype=float)
        candles = pd.DataFrame({
            "open": closes, "high": closes + 0.5, "low": closes - 0.5, "close": closes
        })
        self.assertIsNone(breakout_scanner.momentum_divergence(candles, "CE"))


class BreakoutConfigTests(unittest.TestCase):
    def test_block_over_risk_default_is_true(self):
        """Commit A invariant: risk_warning must be a hard veto, not advisory only."""
        self.assertTrue(breakout_scanner.CONFIG["block_over_risk"])

    def test_commit_b_quality_gates_are_enabled_by_default(self):
        self.assertTrue(breakout_scanner.CONFIG["require_vwap_alignment"])
        self.assertTrue(breakout_scanner.CONFIG["require_prior_day_level"])
        self.assertTrue(breakout_scanner.CONFIG["veto_momentum_divergence"])
        self.assertEqual(breakout_scanner.CONFIG["min_adx"], 20.0)


if __name__ == "__main__":
    unittest.main()
