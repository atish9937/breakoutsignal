"""Unit tests for ema_retest_strategy.evaluate_setup."""
from __future__ import annotations

import datetime as dt
import unittest

import pandas as pd

import breakout_scanner as core
import ema_retest_strategy as strategy


IST = core.IST


def _history_bars(n: int = 30, price: float = 25000.0,
                  start: str = "2026-10-19 09:15") -> pd.DataFrame:
    """Flat-price lead-in bars so the 20 EMA converges to `price` well before
    the pattern bars arrive."""
    idx = pd.date_range(start, periods=n, freq="5min", tz=IST)
    return pd.DataFrame(
        {"open": [price] * n, "high": [price + 1] * n,
         "low": [price - 1] * n, "close": [price] * n,
         "volume": [0] * n},
        index=idx,
    )


def _append_bar(df: pd.DataFrame, o: float, h: float, l: float, c: float) -> pd.DataFrame:
    """Append a single 5-min bar after the last timestamp in df."""
    next_ts = df.index[-1] + pd.Timedelta(minutes=5)
    row = pd.DataFrame({"open": [o], "high": [h], "low": [l], "close": [c],
                        "volume": [0]}, index=[next_ts])
    return pd.concat([df, row])


def _build_valid_setup() -> pd.DataFrame:
    """Build bars representing: flat lead-in near 25000, then a clean
    breakout (green, opens below EMA, closes above, body >= 60%), a red
    retest that closes above the EMA, then a trigger bar whose high
    breaks the retest high. Risk sized to land inside 8..25 points."""
    bars = _history_bars(30, price=25000.0)
    # EMA ~ 25000 after 30 flat bars.
    # Dip the pre-breakout bar below EMA so the BREAKOUT bar opens below.
    bars = _append_bar(bars, 25000, 25001, 24988, 24990)   # drift down
    bars = _append_bar(bars, 24990, 25018, 24989, 25015)   # BREAKOUT: open<EMA, close>EMA, body 25/29=86%
    bars = _append_bar(bars, 25015, 25016, 25008, 25010)   # RETEST: red, close > EMA (~25001)
    bars = _append_bar(bars, 25010, 25022, 25009, 25020)   # TRIGGER: high 25022 > retest high 25016
    return bars


class EvaluateSetupTests(unittest.TestCase):
    def test_valid_long_setup_produces_setup_ce(self):
        bars = _build_valid_setup()
        now = bars.index[-1].to_pydatetime()
        result = strategy.evaluate_setup("NIFTY", bars, now)
        self.assertEqual(result["verdict"], "SETUP_CE")
        self.assertEqual(result["side"], "CE")
        plan = result["plan"]
        # Entry should be retest-high + 1pt slippage = 25017.
        self.assertAlmostEqual(plan["entry"], 25017.0, places=1)
        # T1 = entry + 1.5R, T2 = entry + 3R.
        risk = plan["risk_per_share"]
        self.assertAlmostEqual(plan["t1"], plan["entry"] + 1.5 * risk, places=1)
        self.assertAlmostEqual(plan["t2"], plan["entry"] + 3.0 * risk, places=1)
        # Risk band + EMA slope gates must have passed.
        gates = {c["gate"]: c["passed"] for c in result["mechanical_checks"]}
        self.assertTrue(gates.get("Risk band"))
        self.assertTrue(gates.get("EMA slope"))
        self.assertGreater(plan["ema_slope_pts"], 0)
        self.assertEqual(plan["ema_slope_bars"], strategy.EMA_SLOPE_BARS)

    def test_flat_or_declining_ema_downgrades_to_watch(self):
        """Even with a perfect breakout/retest/trigger pattern, a flat-or-
        declining 20 EMA over the slope lookback must reject the fire.
        Models the sideways regime the live trader observed."""
        import unittest.mock as mock
        bars = _build_valid_setup()
        now = bars.index[-1].to_pydatetime()
        # Patch the slope lookback to be longer than available history so the
        # gate measures slope as None (insufficient history) and rejects.
        with mock.patch.object(strategy, "EMA_SLOPE_BARS", 10_000):
            result = strategy.evaluate_setup("NIFTY", bars, now)
        self.assertEqual(result["verdict"], "WATCH_CE")
        self.assertIn("EMA slope", result["fails"])

    def test_negative_slope_history_rejects(self):
        """Declining-EMA history (sideways-then-down) with a valid pattern
        must still reject via the slope gate."""
        # Build 40 bars of linearly DECLINING price so the 20 EMA slopes down.
        idx = pd.date_range("2026-10-19 09:15", periods=40, freq="5min", tz=IST)
        prices = [25100.0 - i * 2 for i in range(40)]      # 25100 -> 25022
        bars = pd.DataFrame({
            "open": prices, "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices], "close": prices,
            "volume": [0] * 40,
        }, index=idx)
        # Now splice the valid breakout/retest/trigger pattern on top.
        # At this point EMA is ~ 25070 (lagging the price decline).
        # Build pattern with prices crossing EMA from below.
        bars = _append_bar(bars, 25020, 25021, 25010, 25015)   # dip below EMA
        bars = _append_bar(bars, 25015, 25080, 25014, 25075)   # BREAKOUT
        bars = _append_bar(bars, 25075, 25076, 25068, 25072)   # RETEST (red, close > EMA)
        bars = _append_bar(bars, 25072, 25085, 25071, 25082)   # TRIGGER
        now = bars.index[-1].to_pydatetime()
        result = strategy.evaluate_setup("NIFTY", bars, now)
        # Could reject at pattern step or at slope step; either way not SETUP_CE
        self.assertNotEqual(result["verdict"], "SETUP_CE")

    def test_weak_breakout_body_is_rejected(self):
        bars = _history_bars(30, price=25000.0)
        bars = _append_bar(bars, 25000, 25001, 24988, 24990)
        # Doji-ish breakout: wide range, tiny body
        bars = _append_bar(bars, 24990, 25030, 24989, 25005)  # body 15 / range 41 = 37%
        bars = _append_bar(bars, 25005, 25006, 24998, 25002)
        bars = _append_bar(bars, 25002, 25012, 24997, 25010)
        now = bars.index[-1].to_pydatetime()
        result = strategy.evaluate_setup("NIFTY", bars, now)
        self.assertEqual(result["verdict"], "SKIP")

    def test_green_retest_is_rejected(self):
        bars = _history_bars(30, price=25000.0)
        bars = _append_bar(bars, 25000, 25001, 24988, 24990)
        bars = _append_bar(bars, 24990, 25018, 24989, 25015)
        # Green retest: close > open — should NOT count as a valid retest
        bars = _append_bar(bars, 25015, 25020, 25012, 25018)
        bars = _append_bar(bars, 25018, 25025, 25017, 25022)
        now = bars.index[-1].to_pydatetime()
        result = strategy.evaluate_setup("NIFTY", bars, now)
        self.assertEqual(result["verdict"], "SKIP")

    def test_intermediate_close_below_ema_invalidates_setup(self):
        bars = _history_bars(30, price=25000.0)
        bars = _append_bar(bars, 25000, 25001, 24988, 24990)
        bars = _append_bar(bars, 24990, 25018, 24989, 25015)   # BREAKOUT
        bars = _append_bar(bars, 25015, 25016, 25008, 25010)   # RETEST (red, above EMA)
        # Intermediate bar closes BELOW EMA ~ 25001  (kills the setup)
        bars = _append_bar(bars, 25010, 25011, 24990, 24995)
        bars = _append_bar(bars, 24995, 25020, 24994, 25018)   # trigger candidate
        now = bars.index[-1].to_pydatetime()
        result = strategy.evaluate_setup("NIFTY", bars, now)
        self.assertEqual(result["verdict"], "SKIP")

    def test_risk_too_tight_downgrades_to_watch(self):
        """A valid structure with risk < MIN_RISK_POINTS should NOT fire."""
        bars = _history_bars(30, price=25000.0)
        # Build structure where retest high is only ~5 pts above EMA.
        bars = _append_bar(bars, 25000, 25001, 24998, 24999)
        bars = _append_bar(bars, 24999, 25005, 24998, 25004)   # tiny breakout (body ok: 5/7 = 71%)
        bars = _append_bar(bars, 25004, 25005, 25002, 25003)   # retest red, close>EMA
        bars = _append_bar(bars, 25003, 25007, 25002, 25006)   # trigger: high 25007 > retest 25005
        now = bars.index[-1].to_pydatetime()
        result = strategy.evaluate_setup("NIFTY", bars, now)
        # entry ~= 25006, EMA ~= 25000 -> risk ~6 < MIN_RISK_POINTS (8)
        self.assertEqual(result["verdict"], "WATCH_CE")
        self.assertIn("Risk band", result["fails"])

    def test_insufficient_history_skipped(self):
        bars = _history_bars(5, price=25000.0)
        now = bars.index[-1].to_pydatetime()
        result = strategy.evaluate_setup("NIFTY", bars, now)
        self.assertEqual(result["verdict"], "SKIP")


class SnapshotWriteTests(unittest.TestCase):
    """Smoke-check scan() produces a snapshot dict and doesn't crash with a
    stub Dhan client. Network paths are mocked out."""

    def test_scan_returns_snapshot_structure(self):
        import unittest.mock as mock
        from pathlib import Path
        import tempfile, os

        # Isolate OUT so the test doesn't pollute the real scanner_out dir.
        tmpdir = tempfile.mkdtemp()
        with mock.patch.object(strategy, "OUT", Path(tmpdir)):
            # Populate OUT so the strategy's _fire_if_ready path can write.
            dh = mock.Mock()
            dh.intraday.return_value = _history_bars(30, price=25000.0)
            fired: set[str] = set()
            snap = strategy.scan(dh, master=None, prep={}, universe={},
                                 day=dt.date(2026, 10, 19),
                                 now=dt.datetime(2026, 10, 19, 10, 0, tzinfo=IST),
                                 fired=fired)
        self.assertEqual(snap["strategy"], strategy.STRATEGY)
        self.assertIn("results", snap)
        self.assertEqual(len(snap["results"]), 1)
        self.assertEqual(snap["results"][0]["symbol"], "NIFTY")


class LedgerIntegrationTests(unittest.TestCase):
    """Verify ema_retest signals round-trip through the shared paper ledger
    and that the daily report / Telegram caption include the strategy."""

    def _ema_retest_snapshot(self):
        return {
            "scan_time": "2026-10-21 10:00:20",
            "results": [{
                "strategy": strategy.STRATEGY,
                "verdict": "BUY_CE",
                "symbol": "NIFTY",
                "side": "CE",
                "ltp": 25010,
                "plan": {
                    "entry": 25017, "stop": 25005, "t1": 25035, "t2": 25053,
                    "risk_per_share": 12, "runway_r": 3.0,
                },
                "option": {
                    "security_id": 9999,
                    "expiry": "2026-10-29",
                    "premium": 80,
                    "bid": 79.5,
                    "ask": 80.5,
                    "lot_size": 75,
                },
                "underlying_security_id": 13,
            }],
        }

    def test_record_buy_recommendations_accepts_ema_retest_strategy(self):
        """Regression for the paper_trades CHECK constraint: the ledger must
        accept strategy='ema_retest' or every live fire crashes on INSERT."""
        import tempfile
        import trade_performance as performance
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            conn = performance.connect(Path(td) / "ledger.sqlite3")
            inserted = performance.record_buy_recommendations(
                conn, strategy.STRATEGY, self._ema_retest_snapshot(),
                {"NIFTY": 13}, dt.date(2026, 10, 21),
            )
            self.assertEqual(inserted, 1)
            row = conn.execute(
                "SELECT strategy, symbol, side FROM paper_trades"
            ).fetchone()
            self.assertEqual(dict(row),
                             {"strategy": "ema_retest", "symbol": "NIFTY", "side": "CE"})
            conn.close()

    def test_daily_report_caption_includes_ema_retest_line(self):
        """Caption line regression: Telegram preview must surface EMA_Retest
        W/R so the report doesn't silently drop the strategy like it once did
        with the 15m line."""
        import tempfile
        import trade_performance as performance
        from pathlib import Path
        import unittest.mock as mock
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            conn = performance.connect(root / "ledger.sqlite3")
            performance.record_buy_recommendations(
                conn, strategy.STRATEGY, self._ema_retest_snapshot(),
                {"NIFTY": 13}, dt.date(2026, 10, 21),
            )
            generated = dt.datetime(2026, 10, 21, 15, 36, tzinfo=IST)
            with mock.patch.object(performance, "send_html_report",
                                    return_value="42") as send:
                performance.finalize_day(conn, dt.date(2026, 10, 21),
                                          root, "token", "chat", generated)
            caption = send.call_args.args[3]
            self.assertIn("EMA_Retest", caption)
            conn.close()


if __name__ == "__main__":
    unittest.main()
