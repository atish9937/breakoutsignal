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

    def test_strike_walker_is_enabled_by_default(self):
        """Commit B.1 invariant: when ATM breaches the risk cap, walk the strike
        rather than drop the signal. Without this the breakout strategy is
        dead on high-priced NSE names."""
        self.assertTrue(breakout_scanner.CONFIG["walk_strike_on_over_risk"])
        self.assertGreaterEqual(breakout_scanner.CONFIG["max_strike_walks"], 1)
        self.assertEqual(breakout_scanner.CONFIG["option_min_delta"], 0.25)

    def test_commit_g_macd_hard_gate_and_min_t1_distance_defaults(self):
        """Commit G defaults - after SUNPHARMA (MACD -0.32 but RSI OK) and
        ONGC (T1 only 0.23% away) losses on 2026-10-29."""
        self.assertTrue(breakout_scanner.CONFIG["require_macd_momentum"])
        self.assertEqual(breakout_scanner.CONFIG["min_t1_distance_pct"], 0.40)
        # Same gate exists in trend_pullback
        import trend_pullback
        self.assertEqual(trend_pullback.MIN_T1_DISTANCE_PCT, 0.40)


class PriorWeekAndMonthLevelsTests(unittest.TestCase):
    """Commit H: liquidity_sweep now also considers prior week + prior month
    highs/lows as liquidity pools (where swing-trader and institutional stops
    accumulate)."""

    def _daily_frame(self, start: str, periods: int, highs, lows):
        import datetime as dt
        idx = pd.date_range(start, periods=periods, freq="D",
                            tz=dt.timezone(dt.timedelta(hours=5, minutes=30)))
        return pd.DataFrame({
            "open": lows, "high": highs, "low": lows,
            "close": [(h + l) / 2 for h, l in zip(highs, lows)],
            "volume": [0] * periods,
        }, index=idx)

    def test_prior_week_hl_returns_last_completed_week(self):
        import datetime as dt
        # 2026-10-30 is a Fri (ISO week 44). Prior week = 10-19 Mon .. 10-23 Fri (week 43).
        # Build daily frame covering 10-19 .. 10-30.
        daily = self._daily_frame("2026-10-19", 12,
                                   highs=[100, 102, 108, 105, 110, 100, 100, 100, 100, 100, 100, 100],
                                   lows=[90, 92, 93, 94, 95, 90, 90, 90, 90, 90, 90, 90])
        today = dt.date(2026, 10, 30)
        hi, lo = breakout_scanner.prior_week_high_low(daily, today)
        # Week 43 was 2026-10-19 (Mon) through 2026-10-23 (Fri), 5 bars.
        self.assertEqual(hi, 110)  # max of 100,102,108,105,110
        self.assertEqual(lo, 90)   # min of 90,92,93,94,95

    def test_prior_month_hl_returns_last_completed_calendar_month(self):
        import datetime as dt
        # Sep 2026 has highs up to 100; Oct starts with higher prints.
        daily = self._daily_frame("2026-09-01", 42,
                                   highs=[80 + i for i in range(30)] + [999] * 12,
                                   lows=[70 + i for i in range(30)] + [0] * 12)
        today = dt.date(2026, 10, 15)
        hi, lo = breakout_scanner.prior_month_high_low(daily, today)
        # Only Sep 2026 rows count: highs are 80..109, lows 70..99
        self.assertEqual(hi, 109)
        self.assertEqual(lo, 70)


class StrikeWalkerTests(unittest.TestCase):
    """pick_option now exposes walk_candidates so evaluate() can scan OTM strikes."""

    def _chain(self):
        return {
            "oc": {
                "100": {"ce": {"last_price": 10, "top_bid_price": 9.9, "top_ask_price": 10.1,
                               "oi": 2000, "greeks": {"delta": 0.55}}, "pe": {}},
                "105": {"ce": {"last_price": 7,  "top_bid_price": 6.9, "top_ask_price": 7.1,
                               "oi": 1500, "greeks": {"delta": 0.42}}, "pe": {}},
                "110": {"ce": {"last_price": 5,  "top_bid_price": 4.9, "top_ask_price": 5.1,
                               "oi": 1200, "greeks": {"delta": 0.30}}, "pe": {}},
                "115": {"ce": {"last_price": 3,  "top_bid_price": 2.95, "top_ask_price": 3.05,
                               "oi": 1100, "greeks": {"delta": 0.20}}, "pe": {}},
            }
        }

    def test_walk_candidates_lists_otm_strikes_above_atm_for_calls(self):
        info = breakout_scanner.pick_option(self._chain(), "CE", 100.0, 100.0)
        self.assertIsNotNone(info)
        self.assertEqual(info["strike"], 100.0)  # ATM is the closest strike
        walked = [s for s, _ in info["walk_candidates"]]
        self.assertEqual(walked, [105.0, 110.0, 115.0])

    def test_walk_candidates_lists_otm_strikes_below_atm_for_puts(self):
        chain = {
            "oc": {
                "95":  {"pe": {"last_price": 3, "greeks": {"delta": 0.25}}, "ce": {}},
                "100": {"pe": {"last_price": 5, "greeks": {"delta": 0.40}}, "ce": {}},
                "105": {"pe": {"last_price": 8, "greeks": {"delta": 0.55}}, "ce": {}},
            }
        }
        info = breakout_scanner.pick_option(chain, "PE", 105.0, 105.0)
        self.assertEqual(info["strike"], 105.0)
        walked = [s for s, _ in info["walk_candidates"]]
        self.assertEqual(walked, [100.0, 95.0])  # OTM for PE = lower strikes


if __name__ == "__main__":
    unittest.main()
