"""Unit tests for the Opening Range Breakout strategy."""
import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd

import breakout_scanner as core
import opening_range_breakout as strategy


IST = core.IST


def _bars(start: str, closes: list[float], highs: list[float] | None = None,
          lows: list[float] | None = None, volume: int = 1000) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="5min", tz=IST)
    highs = highs if highs is not None else [c + 1 for c in closes]
    lows = lows if lows is not None else [c - 1 for c in closes]
    return pd.DataFrame({
        "open": [c for c in closes],
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [volume] * len(closes),
    }, index=idx)


class OpeningRangeTests(unittest.TestCase):
    def test_opening_range_covers_the_first_30_min(self):
        """OR = high/low of 09:15-09:45 bars only."""
        # 09:15, 09:20, 09:25, 09:30, 09:35, 09:40 are the OR bars (6 x 5-min)
        # 09:45 and later are OUTSIDE the OR
        bars = _bars("2026-11-04 09:15",
                     closes=[100, 102, 105, 103, 101, 104,  # 09:15..09:40 (OR)
                             110, 112, 108],                 # 09:45, 09:50, 09:55
                     highs= [101, 103, 108, 104, 103, 106,
                             115, 118, 113],
                     lows=  [99,  101, 103, 101, 99,  102,
                             108, 110, 106])
        or_high, or_low = strategy._opening_range(bars)
        self.assertEqual(or_high, 108)  # max OR bar high
        self.assertEqual(or_low, 99)    # min OR bar low


class ORBBreakoutDetectionTests(unittest.TestCase):
    def _candles_with_or_and_post(self, side="CE"):
        # OR: 09:15-09:40 (6 bars), consolidating in [99, 108]
        or_closes = [102, 104, 106, 103, 101, 105]
        or_highs =  [103, 105, 108, 104, 102, 106]
        or_lows =   [100, 102, 103, 102, 99,  104]
        if side == "CE":
            # Breakout bar at 09:45: close 113 (well above or_high=108)
            post_closes = [113]
            post_highs =  [114]
            post_lows =   [107]
            post_volumes = [2000]  # high volume
        else:
            # Breakout bar at 09:45: close 95 (well below or_low=99)
            post_closes = [95]
            post_highs =  [100]
            post_lows =   [94]
            post_volumes = [2000]
        closes = or_closes + post_closes
        highs = or_highs + post_highs
        lows = or_lows + post_lows
        volumes = [1000] * 6 + post_volumes
        idx = pd.date_range("2026-11-04 09:15", periods=len(closes), freq="5min", tz=IST)
        return pd.DataFrame({
            "open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes,
        }, index=idx)

    def test_bullish_or_breakout_detected(self):
        candles = self._candles_with_or_and_post(side="CE")
        now = candles.index[-1].to_pydatetime() + dt.timedelta(seconds=20)
        with patch.object(core, "atr", return_value=1.0), \
                patch.object(core, "adx", return_value=25.0), \
                patch.object(core, "macd_hist", return_value=0.5):
            result = strategy.evaluate_setup(
                "TEST", candles, now,
                prev_high=109.0, prev_low=99.0,
                nifty_bias={"bias": "UP", "chg_pct": 0.4},
                vix={"ltp": 15.0, "ema20": 15.0},
                breadth={"pct_above_prev_close": 65.0},
            )
        self.assertEqual(result["side"], "CE")
        self.assertEqual(result["verdict"], "SETUP_CE")
        self.assertEqual(result["or_high"], 108)
        self.assertEqual(result["or_low"], 99)

    def test_bearish_or_breakout_detected(self):
        candles = self._candles_with_or_and_post(side="PE")
        now = candles.index[-1].to_pydatetime() + dt.timedelta(seconds=20)
        with patch.object(core, "atr", return_value=1.0), \
                patch.object(core, "adx", return_value=25.0), \
                patch.object(core, "macd_hist", return_value=-0.5):
            # prev_low=96 so breakout close 95 is BELOW it (passes prior-day H/L gate)
            result = strategy.evaluate_setup(
                "TEST", candles, now,
                prev_high=108.0, prev_low=96.0,
                nifty_bias={"bias": "DOWN", "chg_pct": -0.5},
                vix={"ltp": 15.0, "ema20": 15.0},
                breadth={"pct_above_prev_close": 35.0},
            )
        self.assertEqual(result["side"], "PE")
        self.assertEqual(result["verdict"], "SETUP_PE")

    def test_inside_or_bar_rejected(self):
        """A bar whose close stays INSIDE the OR must be SKIP."""
        # Breakout bar at 09:45 closes at 103 - still inside OR [99, 108]
        or_closes = [102, 104, 106, 103, 101, 105]
        post_closes = [103]
        closes = or_closes + post_closes
        idx = pd.date_range("2026-11-04 09:15", periods=len(closes), freq="5min", tz=IST)
        df = pd.DataFrame({
            "open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
            "close": closes, "volume": [1000] * len(closes),
        }, index=idx)
        now = idx[-1].to_pydatetime() + dt.timedelta(seconds=20)
        with patch.object(core, "atr", return_value=1.0), \
                patch.object(core, "adx", return_value=25.0):
            result = strategy.evaluate_setup(
                "TEST", df, now,
                prev_high=109.0, prev_low=99.0,
                nifty_bias={"bias": "FLAT", "chg_pct": 0.0},
            )
        self.assertEqual(result["verdict"], "SKIP")

    def test_entry_window_gate_blocks_after_11am(self):
        """A breakout bar closing at 11:05 is OUTSIDE the entry window -> SKIP."""
        or_closes = [102, 104, 106, 103, 101, 105]
        # Need 23 bars: 6 OR (09:15-09:40) + 17 post (09:45-11:05)
        closes = or_closes + [106] * 16 + [113]
        n = len(closes)
        idx = pd.date_range("2026-11-04 09:15", periods=n, freq="5min", tz=IST)
        highs = [c + 1 for c in closes]
        highs[-1] = 115
        lows = [c - 1 for c in closes]
        df = pd.DataFrame({
            "open": closes, "high": highs, "low": lows, "close": closes,
            "volume": [1000] * (n - 1) + [2000],
        }, index=idx)
        now = idx[-1].to_pydatetime() + dt.timedelta(seconds=20)
        self.assertGreater(idx[-1].time(), strategy.ENTRY_END)  # 11:05 > 11:00
        with patch.object(core, "atr", return_value=1.0):
            result = strategy.evaluate_setup(
                "TEST", df, now,
                prev_high=109.0, prev_low=99.0,
                nifty_bias={"bias": "UP", "chg_pct": 0.4},
            )
        self.assertEqual(result["verdict"], "SKIP")
        self.assertIn("outside entry window", result["fails"])


class ORBConfigTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(strategy.STRATEGY, "opening_range_breakout")
        self.assertEqual(strategy.OR_START, dt.time(9, 15))
        self.assertEqual(strategy.OR_END, dt.time(9, 45))
        self.assertEqual(strategy.ENTRY_END, dt.time(11, 0))
        self.assertEqual(strategy.MIN_PENETRATION_ATR, 0.20)
        self.assertEqual(strategy.MIN_VOLUME_RATIO, 1.5)
        self.assertEqual(strategy.MAX_FIRES_PER_DAY, 2)
        # Option gates match the other strategies
        self.assertEqual(strategy.OPTION_MIN_DELTA, 0.25)
        self.assertEqual(strategy.OPTION_MIN_DTE_DAYS, 5)
        self.assertEqual(strategy.MIN_T1_DISTANCE_PCT, 0.40)


class IndexSupportTests(unittest.TestCase):
    """Commit N: ORB now scans NIFTY + BANKNIFTY alongside the stock universe.
    Reuses liquidity_sweep's INDICES config so index pool/expiry resolution
    stays consistent across both strategies."""

    def test_orb_reads_indices_from_liquidity_sweep(self):
        """ORB intentionally has no local INDICES config - the index iteration
        inside scan() imports and uses liquidity_sweep.INDICES. This test
        pins that contract so a future refactor can't silently drop index
        support."""
        import liquidity_sweep as ls
        self.assertIn("NIFTY", ls.INDICES)
        self.assertIn("BANKNIFTY", ls.INDICES)
        self.assertEqual(ls.INDICES["NIFTY"]["sid"], 13)
        self.assertEqual(ls.INDICES["BANKNIFTY"]["sid"], 25)

    def test_orb_scan_body_references_indices_block(self):
        """Guard against the index block being accidentally removed from scan()."""
        import pathlib
        src = pathlib.Path("opening_range_breakout.py").read_text(encoding="utf-8")
        # Index iteration must reference liquidity_sweep.INDICES
        self.assertIn("_ls.INDICES.items()", src,
                      "ORB.scan() must iterate liquidity_sweep.INDICES for index coverage")
        # Must stamp underlying_security_id on index results so ledger insert works
        self.assertIn('result["underlying_security_id"] = sid', src)


class RoundNumberHelperTests(unittest.TestCase):
    def test_tiers(self):
        # <500 -> step 10
        self.assertEqual(strategy._nearest_round_numbers(452)[1], 450)
        self.assertEqual(strategy._nearest_round_numbers(452)[0], 460)
        # 500-5000 -> step 50
        self.assertEqual(strategy._nearest_round_numbers(1218)[1], 1200)
        self.assertEqual(strategy._nearest_round_numbers(1218)[0], 1250)
        # >5000 -> step 100
        self.assertEqual(strategy._nearest_round_numbers(5250)[1], 5200)
        self.assertEqual(strategy._nearest_round_numbers(5250)[0], 5300)


if __name__ == "__main__":
    unittest.main()
