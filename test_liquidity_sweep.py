"""Unit tests for the liquidity-sweep strategy."""
import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd

import breakout_scanner as core
import liquidity_sweep as strategy


IST = core.IST


def _flat_bars(n: int, price: float = 100.0, start: str = "2026-10-19 09:30",
               freq: str = "5min") -> pd.DataFrame:
    """n bars of near-flat price - used as the "normal" context before a pivot."""
    idx = pd.date_range(start, periods=n, freq=freq, tz=IST)
    df = pd.DataFrame(
        {"open": [price] * n, "high": [price + 0.5] * n,
         "low": [price - 0.5] * n, "close": [price] * n,
         "volume": [1000] * n},
        index=idx,
    )
    return df


class PivotDetectionTests(unittest.TestCase):
    def test_pivot_high_requires_strict_dominance_in_window(self):
        bars = _flat_bars(10)
        # Make bar 5 a clean pivot high
        bars.loc[bars.index[5], "high"] = 110.0
        highs, lows = strategy._find_pivots(bars, width=2)
        self.assertIn(110.0, highs)

    def test_pivot_low_requires_strict_dominance_in_window(self):
        bars = _flat_bars(10)
        bars.loc[bars.index[5], "low"] = 90.0
        highs, lows = strategy._find_pivots(bars, width=2)
        self.assertIn(90.0, lows)

    def test_edge_bars_are_never_pivots(self):
        bars = _flat_bars(10)
        bars.loc[bars.index[0], "high"] = 115.0
        bars.loc[bars.index[-1], "low"] = 85.0
        highs, lows = strategy._find_pivots(bars, width=2)
        self.assertNotIn(115.0, highs)
        self.assertNotIn(85.0, lows)

    def test_tied_highs_in_window_are_not_pivots(self):
        """A pivot must be the UNIQUE max/min inside the window."""
        bars = _flat_bars(10)
        bars.loc[bars.index[5], "high"] = 110.0
        bars.loc[bars.index[6], "high"] = 110.0  # tie - neither can be a pivot
        highs, _ = strategy._find_pivots(bars, width=2)
        self.assertNotIn(110.0, highs)


class WickRatioTests(unittest.TestCase):
    def test_upper_wick_dominant_bar(self):
        bar = pd.Series({"open": 100, "high": 110, "low": 99, "close": 101})
        upper, lower = strategy._wick_ratios(bar)
        self.assertAlmostEqual(upper, (110 - 101) / 11, places=3)
        self.assertAlmostEqual(lower, (100 - 99) / 11, places=3)

    def test_zero_range_bar_returns_zero(self):
        bar = pd.Series({"open": 100, "high": 100, "low": 100, "close": 100})
        self.assertEqual(strategy._wick_ratios(bar), (0.0, 0.0))


class SweepDetectionTests(unittest.TestCase):
    """Direct unit tests of _detect_sweep against hand-crafted bars."""

    def _prev_bars(self, bars: int = 20) -> pd.DataFrame:
        """20 flat-volume bars to establish the pre-sweep baseline."""
        df = _flat_bars(bars)
        df["volume"] = 1000
        return df

    def test_bearish_sweep_above_pool_high_is_detected(self):
        prev = self._prev_bars()
        atr = 1.0
        pool_high = 110.0
        # Sweep bar: high pokes above pool_high, closes back below, big upper wick, big volume
        sweep_bar = pd.Series({
            "open": 109.5, "high": 112.0, "low": 109.0, "close": 109.5, "volume": 2000
        })
        info = strategy._detect_sweep(sweep_bar, prev, [pool_high], [90.0], atr)
        self.assertIsNotNone(info)
        self.assertEqual(info["side"], "PE")
        self.assertAlmostEqual(info["pool_level"], 110.0)
        self.assertAlmostEqual(info["sweep_extreme"], 112.0)

    def test_bullish_sweep_below_pool_low_is_detected(self):
        prev = self._prev_bars()
        atr = 1.0
        pool_low = 90.0
        sweep_bar = pd.Series({
            "open": 90.5, "high": 91.0, "low": 88.0, "close": 90.5, "volume": 2000
        })
        info = strategy._detect_sweep(sweep_bar, prev, [110.0], [pool_low], atr)
        self.assertIsNotNone(info)
        self.assertEqual(info["side"], "CE")
        self.assertAlmostEqual(info["pool_level"], 90.0)
        self.assertAlmostEqual(info["sweep_extreme"], 88.0)

    def test_sweep_without_close_back_inside_is_rejected(self):
        """Price breaks above pool and keeps going - a breakout, not a sweep."""
        prev = self._prev_bars()
        atr = 1.0
        sweep_bar = pd.Series({
            "open": 109.5, "high": 112.0, "low": 109.0, "close": 111.5, "volume": 2000
        })
        self.assertIsNone(strategy._detect_sweep(sweep_bar, prev, [110.0], [90.0], atr))

    def test_sweep_without_dominant_wick_is_rejected(self):
        """Price pokes above and reclaims but the wick is small - no stop hunt signature."""
        prev = self._prev_bars()
        atr = 1.0
        # Price pokes to 110.6, closes inside at 109.9. Wide range below pool
        # keeps the upper wick at only ~27% of total range - not a sweep wick.
        sweep_bar = pd.Series({
            "open": 109.0, "high": 110.6, "low": 108.0, "close": 109.9, "volume": 2000
        })
        self.assertIsNone(strategy._detect_sweep(sweep_bar, prev, [110.0], [90.0], atr))

    def test_sweep_without_volume_is_rejected(self):
        prev = self._prev_bars()
        atr = 1.0
        sweep_bar = pd.Series({
            "open": 109.5, "high": 112.0, "low": 109.0, "close": 109.5, "volume": 1000  # avg volume
        })
        self.assertIsNone(strategy._detect_sweep(sweep_bar, prev, [110.0], [90.0], atr))

    def test_sweep_without_sufficient_penetration_is_rejected(self):
        """Pokes above by less than SWEEP_PENETRATION_ATR * ATR - noise, not a sweep."""
        prev = self._prev_bars()
        atr = 5.0  # requires >= 0.5 points penetration
        sweep_bar = pd.Series({
            # pokes by only 0.1 pts above pool_high
            "open": 109.5, "high": 110.1, "low": 108.0, "close": 109.5, "volume": 2000
        })
        self.assertIsNone(strategy._detect_sweep(sweep_bar, prev, [110.0], [90.0], atr))


class ConfirmationBarTests(unittest.TestCase):
    def test_bearish_sweep_needs_strong_down_close(self):
        sweep_bar = pd.Series({"open": 109.5, "high": 112.0, "low": 109.0, "close": 109.5})
        atr = 1.0
        # Good bearish confirmation: big body, close near low, below sweep close
        confirm = pd.Series({"open": 109.5, "high": 109.5, "low": 108.0, "close": 108.2})
        ok, _ = strategy._confirms(confirm, sweep_bar, "PE", atr)
        self.assertTrue(ok)

    def test_bearish_sweep_rejects_small_body_confirmation(self):
        sweep_bar = pd.Series({"open": 109.5, "high": 112.0, "low": 109.0, "close": 109.5})
        atr = 10.0  # requires body >= 5
        confirm = pd.Series({"open": 109.5, "high": 109.5, "low": 108.0, "close": 108.5})
        ok, measurements = strategy._confirms(confirm, sweep_bar, "PE", atr)
        self.assertFalse(ok)
        self.assertIn("body", measurements["reason"])

    def test_bearish_sweep_rejects_weak_close_position(self):
        """Close in the middle of the bar isn't convicted enough."""
        sweep_bar = pd.Series({"open": 109.5, "high": 112.0, "low": 109.0, "close": 109.5})
        atr = 1.0
        # body is big but close is in the middle of the range
        confirm = pd.Series({"open": 110.0, "high": 110.0, "low": 108.0, "close": 109.0})
        ok, measurements = strategy._confirms(confirm, sweep_bar, "PE", atr)
        self.assertFalse(ok)
        self.assertIn("close", measurements["reason"])

    def test_bearish_sweep_rejects_close_not_below_sweep_close(self):
        sweep_bar = pd.Series({"open": 109.5, "high": 112.0, "low": 109.0, "close": 109.5})
        atr = 1.0
        # body big, close near low, but close > sweep close
        confirm = pd.Series({"open": 110.0, "high": 110.5, "low": 109.7, "close": 109.75})
        ok, _ = strategy._confirms(confirm, sweep_bar, "PE", atr)
        self.assertFalse(ok)

    def test_bullish_sweep_confirms_with_strong_up_close(self):
        sweep_bar = pd.Series({"open": 90.5, "high": 91.0, "low": 88.0, "close": 90.5})
        atr = 1.0
        confirm = pd.Series({"open": 90.5, "high": 92.0, "low": 90.5, "close": 91.9})
        ok, _ = strategy._confirms(confirm, sweep_bar, "CE", atr)
        self.assertTrue(ok)


class EvaluateSetupTests(unittest.TestCase):
    """End-to-end: feed a candle series and expect SETUP_PE for a bearish sweep + confirm."""

    def _make_bearish_sweep_series(self, prev_high: float = 110.0) -> pd.DataFrame:
        # 25 flat bars + 1 sweep bar + 1 confirmation bar
        prev = _flat_bars(25)
        prev["volume"] = 1000
        # Insert a pool-high around 110 somewhere earlier to be picked up as a pivot
        prev.loc[prev.index[10], "high"] = 110.0
        # Sweep bar: high 112, close 109.5
        sweep_idx = prev.index[-1] + pd.Timedelta(minutes=5)
        confirm_idx = sweep_idx + pd.Timedelta(minutes=5)
        sweep = pd.DataFrame({
            "open": [109.5], "high": [112.0], "low": [109.0],
            "close": [109.5], "volume": [2000]
        }, index=[sweep_idx])
        confirm = pd.DataFrame({
            "open": [109.5], "high": [109.6], "low": [107.5],
            "close": [107.8], "volume": [2000]
        }, index=[confirm_idx])
        return pd.concat([prev, sweep, confirm])

    def test_full_bearish_sweep_passes_to_setup_pe(self):
        candles = self._make_bearish_sweep_series()
        now = candles.index[-1].to_pydatetime() + dt.timedelta(seconds=20)
        with patch.object(core, "atr", return_value=1.0), \
                patch.object(core, "adx", return_value=25.0):
            result = strategy.evaluate_setup(
                "TEST", candles, now,
                prev_high=110.0, prev_low=90.0,
                nifty_bias={"bias": "DOWN"},
                vix={"ltp": 15.0, "ema20": 15.0},
            )
        self.assertEqual(result["side"], "PE")
        self.assertEqual(result["verdict"], "SETUP_PE")
        self.assertIn("sweep", result)
        self.assertIn("confirmation", result)

    def test_flat_market_produces_no_sweep(self):
        """Pure sideways price action should not trigger a sweep."""
        candles = _flat_bars(30)
        candles["volume"] = 1000
        now = candles.index[-1].to_pydatetime() + dt.timedelta(seconds=20)
        with patch.object(core, "atr", return_value=1.0), \
                patch.object(core, "adx", return_value=15.0):
            result = strategy.evaluate_setup(
                "TEST", candles, now,
                prev_high=120.0, prev_low=80.0,
                nifty_bias={"bias": "FLAT"},
                vix={"ltp": 15.0, "ema20": 15.0},
            )
        self.assertEqual(result["verdict"], "SKIP")
        self.assertIn("no sweep bar", result["fails"])


class ConfigConsistencyTests(unittest.TestCase):
    def test_option_gates_match_other_strategies(self):
        """Keep the option gates in lock-step with Commits A/B/B.1."""
        self.assertEqual(strategy.OPTION_MIN_PREMIUM, 2.0)
        self.assertEqual(strategy.OPTION_MIN_OI, 1000)
        self.assertEqual(strategy.OPTION_MIN_DELTA, 0.25)
        self.assertEqual(strategy.OPTION_MIN_DTE_DAYS, 5)
        self.assertTrue(strategy.OPTION_REQUIRE_BID)
        self.assertEqual(strategy.MIN_ADX, 20.0)


class IndexSupportTests(unittest.TestCase):
    """Commit D: NIFTY + BANKNIFTY are scanned alongside the stock universe."""

    def test_indices_config_contains_nifty_and_banknifty_on_idx_i(self):
        self.assertIn("NIFTY", strategy.INDICES)
        self.assertIn("BANKNIFTY", strategy.INDICES)
        self.assertEqual(strategy.INDICES["NIFTY"]["sid"], 13)
        self.assertEqual(strategy.INDICES["NIFTY"]["underlying_seg"], "IDX_I")
        self.assertEqual(strategy.INDICES["BANKNIFTY"]["sid"], 25)
        self.assertEqual(strategy.INDICES["BANKNIFTY"]["underlying_seg"], "IDX_I")

    def test_pick_index_expiry_filters_by_min_dte(self):
        class _DhStub:
            def expiries(self, sid, seg="NSE_EQ"):
                return ["2026-10-30", "2026-11-06", "2026-11-13", "2026-11-27"]
        day = dt.date(2026, 10, 29)
        # with min_dte=5 we need expiry >= 2026-11-03
        picked = strategy._pick_index_expiry(_DhStub(), 13, "IDX_I", day, 5)
        self.assertEqual(picked, "2026-11-06")

    def test_pick_index_expiry_returns_none_when_no_expiry_fits_dte_floor(self):
        class _DhStub:
            def expiries(self, sid, seg="NSE_EQ"):
                return ["2026-10-30"]  # too close
        day = dt.date(2026, 10, 29)
        self.assertIsNone(strategy._pick_index_expiry(_DhStub(), 13, "IDX_I", day, 5))

    def test_index_prev_high_low_returns_prior_session_only(self):
        """Should not use TODAY's partial high/low as the prev-day reference."""
        day = dt.date(2026, 10, 29)
        idx = pd.date_range("2026-10-27", periods=3, freq="D", tz=IST)
        daily = pd.DataFrame(
            {"open": [100]*3, "high": [110, 115, 120], "low": [95, 93, 97],
             "close": [105, 108, 112], "volume": [0]*3},
            index=idx,
        )

        class _DhStub:
            def daily(self, sid, seg, inst, days):
                return daily
        prev_high, prev_low = strategy._index_prev_high_low(_DhStub(), 13, "IDX_I", day)
        # 10-29 is "today"; the prior session is 10-28 (idx[-2] = Oct 28)
        self.assertEqual(prev_high, 115)
        self.assertEqual(prev_low, 93)

    def test_core_lot_fallback_covers_indices(self):
        """Ensures lot_and_id falls back gracefully for NIFTY/BANKNIFTY."""
        self.assertEqual(core.LOT_FALLBACK.get("NIFTY"), 25)
        self.assertEqual(core.LOT_FALLBACK.get("BANKNIFTY"), 15)

    def test_index_prev_high_low_caches_per_day(self):
        """A 60-cycle session must only hit dh.daily once per index per day."""
        day = dt.date(2026, 10, 29)
        idx = pd.date_range("2026-10-27", periods=3, freq="D", tz=IST)
        daily = pd.DataFrame(
            {"open": [100]*3, "high": [110, 115, 120], "low": [95, 93, 97],
             "close": [105, 108, 112], "volume": [0]*3},
            index=idx,
        )

        calls = []

        class _CountingStub:
            def daily(self, sid, seg, inst, days):
                calls.append(sid)
                return daily
        # clear the cache first so results are deterministic
        strategy._index_prev_hl_cache.clear()
        stub = _CountingStub()
        first = strategy._index_prev_high_low(stub, 13, "IDX_I", day)
        second = strategy._index_prev_high_low(stub, 13, "IDX_I", day)
        third = strategy._index_prev_high_low(stub, 13, "IDX_I", day)
        self.assertEqual(len(calls), 1, "second and third fetches should hit the cache")
        self.assertEqual(first, second)
        self.assertEqual(second, third)


if __name__ == "__main__":
    unittest.main()
