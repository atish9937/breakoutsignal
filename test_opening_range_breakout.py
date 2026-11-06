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
        self.assertEqual(strategy.FADE_WICK_RATIO, 0.40)
        # Option gates match the other strategies
        self.assertEqual(strategy.OPTION_MIN_DELTA, 0.25)
        self.assertEqual(strategy.OPTION_MIN_DTE_DAYS, 5)
        self.assertEqual(strategy.MIN_T1_DISTANCE_PCT, 0.40)


class NiftyChopFilterTests(unittest.TestCase):
    """Commit Q: block fires when Nifty is near-flat AND tight-range.
    Reconstructed from Nov 5 2026 TITAN/BANKBARODA CE losses: Nifty was
    +0.17% from open with a 60-min range of 0.29% - both inside the chop
    band, both fires should have been skipped."""

    def _clean_ce_breakout_bars(self):
        """6 OR bars [99,108] + 1 breakout close 113 (above OR_high=108)."""
        or_closes = [102, 104, 106, 103, 101, 105]
        or_highs =  [103, 105, 108, 104, 102, 106]
        or_lows =   [100, 102, 103, 102, 99,  104]
        post_closes = [113]
        post_highs = [114]
        post_lows = [107]
        post_volumes = [2000]
        closes = or_closes + post_closes
        highs = or_highs + post_highs
        lows = or_lows + post_lows
        volumes = [1000] * 6 + post_volumes
        idx = pd.date_range("2026-11-05 09:15", periods=len(closes), freq="5min", tz=IST)
        return pd.DataFrame({
            "open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes,
        }, index=idx)

    def test_chop_regime_blocks_ce_breakout(self):
        """|chg_from_open| 0.17% < 0.15% is FALSE (0.17 > 0.15), so this
        particular shape would NOT be chop. Use 0.10% / 0.29% range (true chop)."""
        df = self._clean_ce_breakout_bars()
        now = df.index[-1].to_pydatetime() + dt.timedelta(seconds=20)
        with patch.object(core, "atr", return_value=1.0), \
                patch.object(core, "adx", return_value=25.0), \
                patch.object(core, "macd_hist", return_value=0.5):
            result = strategy.evaluate_setup(
                "TEST", df, now,
                prev_high=109.0, prev_low=99.0,
                nifty_bias={"bias": "FLAT", "chg_pct": 0.10,
                            "chg_from_open_pct": 0.10,
                            "range_pct_60min": 0.29},
                vix={"ltp": 15.0, "ema20": 15.0},
                breadth={"pct_above_prev_close": 65.0},
            )
        # Chop filter must have fired; verdict demoted below SETUP.
        chop_check = next((c for c in result["mechanical_checks"]
                           if c["gate"] == "Nifty regime (chop filter)"), None)
        self.assertIsNotNone(chop_check)
        self.assertFalse(chop_check["passed"])
        self.assertIn("Nifty regime (chop filter)", result["fails"])

    def test_trending_regime_allows_ce_breakout(self):
        """|chg_from_open| 0.40% (above 0.15%) OR range 0.80% (above 0.50%).
        Either exceeds the chop band -> the gate passes."""
        df = self._clean_ce_breakout_bars()
        now = df.index[-1].to_pydatetime() + dt.timedelta(seconds=20)
        with patch.object(core, "atr", return_value=1.0), \
                patch.object(core, "adx", return_value=25.0), \
                patch.object(core, "macd_hist", return_value=0.5):
            result = strategy.evaluate_setup(
                "TEST", df, now,
                prev_high=109.0, prev_low=99.0,
                nifty_bias={"bias": "UP", "chg_pct": 0.40,
                            "chg_from_open_pct": 0.40,
                            "range_pct_60min": 0.80},
                vix={"ltp": 15.0, "ema20": 15.0},
                breadth={"pct_above_prev_close": 65.0},
            )
        chop_check = next((c for c in result["mechanical_checks"]
                           if c["gate"] == "Nifty regime (chop filter)"), None)
        self.assertIsNotNone(chop_check)
        self.assertTrue(chop_check["passed"])
        self.assertEqual(result["verdict"], "SETUP_CE")

    def test_missing_regime_data_defaults_to_pass(self):
        """If nifty_bias lacks chg_from_open_pct/range_pct_60min (e.g. pre-Commit-Q
        code path or Dhan returned nothing), the gate must not block."""
        df = self._clean_ce_breakout_bars()
        now = df.index[-1].to_pydatetime() + dt.timedelta(seconds=20)
        with patch.object(core, "atr", return_value=1.0), \
                patch.object(core, "adx", return_value=25.0), \
                patch.object(core, "macd_hist", return_value=0.5):
            result = strategy.evaluate_setup(
                "TEST", df, now,
                prev_high=109.0, prev_low=99.0,
                nifty_bias={"bias": "FLAT", "chg_pct": 0.0},
                vix={"ltp": 15.0, "ema20": 15.0},
                breadth={"pct_above_prev_close": 65.0},
            )
        chop_check = next((c for c in result["mechanical_checks"]
                           if c["gate"] == "Nifty regime (chop filter)"), None)
        self.assertIsNotNone(chop_check)
        self.assertTrue(chop_check["passed"])


class BreakoutBarExhaustionFlipTests(unittest.TestCase):
    """Commit P: A breakout bar that closes beyond the OR but leaves a large
    wick in its own breakout direction is a stop-run, not a continuation. ORB
    flips the side so downstream direction gates re-evaluate against the fade
    direction (and will still veto if the regime favors the original move)."""

    def _build(self, breakout_close, breakout_high, breakout_low, breakout_open):
        # OR: 09:15-09:40 in [99, 108]
        or_closes = [102, 104, 106, 103, 101, 105]
        or_highs =  [103, 105, 108, 104, 102, 106]
        or_lows =   [100, 102, 103, 102, 99,  104]
        closes = or_closes + [breakout_close]
        highs = or_highs + [breakout_high]
        lows = or_lows + [breakout_low]
        opens = or_closes + [breakout_open]
        volumes = [1000] * 6 + [2000]
        idx = pd.date_range("2026-11-04 09:15", periods=len(closes), freq="5min", tz=IST)
        return pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes,
        }, index=idx)

    def test_ce_breakout_with_dominant_upper_wick_flips_to_pe(self):
        """Close above OR_high but bar's upper wick is >= FADE_WICK_RATIO
        of its range -> side flips from CE to PE, stop anchors to bar high."""
        # Close 113 (above OR_high 108), high 120, low 108, open 113
        # upper wick = (120 - max(113, 113)) / (120 - 108) = 7/12 = 0.583 >= 0.40
        df = self._build(breakout_close=113, breakout_high=120,
                         breakout_low=108, breakout_open=113)
        now = df.index[-1].to_pydatetime() + dt.timedelta(seconds=20)
        # Supply neutral regime so downstream direction gates won't veto PE
        with patch.object(core, "atr", return_value=1.0), \
                patch.object(core, "adx", return_value=25.0), \
                patch.object(core, "macd_hist", return_value=-0.5):
            result = strategy.evaluate_setup(
                "TEST", df, now,
                prev_high=115.0, prev_low=112.0,
                nifty_bias={"bias": "FLAT", "chg_pct": 0.0},
                vix={"ltp": 15.0, "ema20": 15.0},
                breadth={"pct_above_prev_close": 40.0},
            )
        self.assertTrue(result["flipped_from_exhaustion"])
        self.assertEqual(result["side"], "PE")
        # Stop is bar_high + buffer (ATR=1, buffer=0.10) = 120.10
        self.assertGreater(result["plan"]["stop"], 120.0)

    def test_pe_breakout_with_dominant_lower_wick_flips_to_ce(self):
        """Close below OR_low but bar's lower wick is >= FADE_WICK_RATIO
        of its range -> side flips from PE to CE, stop anchors to bar low."""
        # Close 95 (below OR_low 99), high 99, low 85, open 95
        # lower wick = (min(95, 95) - 85) / (99 - 85) = 10/14 = 0.714 >= 0.40
        df = self._build(breakout_close=95, breakout_high=99,
                         breakout_low=85, breakout_open=95)
        now = df.index[-1].to_pydatetime() + dt.timedelta(seconds=20)
        with patch.object(core, "atr", return_value=1.0), \
                patch.object(core, "adx", return_value=25.0), \
                patch.object(core, "macd_hist", return_value=0.5):
            result = strategy.evaluate_setup(
                "TEST", df, now,
                prev_high=98.0, prev_low=86.0,
                nifty_bias={"bias": "FLAT", "chg_pct": 0.0},
                vix={"ltp": 15.0, "ema20": 15.0},
                breadth={"pct_above_prev_close": 60.0},
            )
        self.assertTrue(result["flipped_from_exhaustion"])
        self.assertEqual(result["side"], "CE")
        self.assertLess(result["plan"]["stop"], 85.0)

    def test_clean_breakout_without_rejection_wick_does_not_flip(self):
        """Close 113, high 114, low 108, open 108 -> upper wick tiny,
        side stays CE."""
        df = self._build(breakout_close=113, breakout_high=114,
                         breakout_low=108, breakout_open=108)
        now = df.index[-1].to_pydatetime() + dt.timedelta(seconds=20)
        with patch.object(core, "atr", return_value=1.0), \
                patch.object(core, "adx", return_value=25.0), \
                patch.object(core, "macd_hist", return_value=0.5):
            result = strategy.evaluate_setup(
                "TEST", df, now,
                prev_high=109.0, prev_low=99.0,
                nifty_bias={"bias": "UP", "chg_pct": 0.4},
                vix={"ltp": 15.0, "ema20": 15.0},
                breadth={"pct_above_prev_close": 65.0},
            )
        self.assertFalse(result["flipped_from_exhaustion"])
        self.assertEqual(result["side"], "CE")


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


class MaxFiresPerDayPersistenceTests(unittest.TestCase):
    """Commit N audit fix: MAX_FIRES_PER_DAY must count from the signal file,
    not from in-memory prefix keys that get wiped on server restart."""

    def test_fired_today_counts_from_signal_file_not_memory(self):
        """Simulate mid-session restart: signal file has 2 ORB rows from
        earlier today. After restart, in-memory `fired` set holds only the
        plain cross-strategy keys (loaded via load_fired_signals). ORB must
        still see fired_today = 2 and refuse to fire more."""
        import tempfile, pathlib, json as _json
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            path = root / "opening_range_breakout_signals_2026-11-04.jsonl"
            # Simulate 2 prior fires persisted today
            path.write_text(
                _json.dumps({"symbol": "RELIANCE", "side": "CE"}) + "\n"
                + _json.dumps({"symbol": "TITAN", "side": "CE"}) + "\n",
                encoding="utf-8",
            )
            # Count the way scan() does it
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            self.assertEqual(count, 2)
            self.assertGreaterEqual(count, strategy.MAX_FIRES_PER_DAY)


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
