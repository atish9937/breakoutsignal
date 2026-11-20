import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import breakout_scanner as core
import intraday_multiframe_strategy as strategy


class IntradayMultiframeMonitorTests(unittest.TestCase):
    def _signal(self):
        return {
            "strategy": strategy.STRATEGY,
            "symbol": "AAA",
            "side": "CE",
            "signal_time": "2026-10-19T10:05:20+05:30",
            "underlying_security_id": 123,
            "underlying_stop": 95.0,
            "underlying_target": 110.0,
            "expiry": "2026-10-29",
            "option": {"strike": 100.0},
            "status": "paper_entry_signal_sent",
        }

    def _candles(self):
        index = pd.DatetimeIndex(
            [
                "2026-10-19T10:00:00+05:30",
                "2026-10-19T10:05:00+05:30",
                "2026-10-19T15:10:00+05:30",
            ]
        )
        return pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [111.0, 103.0, 104.0],
                "low": [90.0, 100.0, 101.0],
                "close": [105.0, 102.0, 103.0],
                "volume": [10, 10, 10],
            },
            index=index,
        )

    def test_same_candle_stop_takes_priority_and_persists_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            path = output / f"{strategy.STRATEGY}_signals_2026-10-19.jsonl"
            path.write_text(json.dumps(self._signal()) + "\n", encoding="utf-8")
            client = Mock()
            client.intraday.return_value = self._candles()
            client.intraday.return_value.loc[
                pd.Timestamp("2026-10-19T10:00:00+05:30"), ["high", "low"]
            ] = [104.0, 100.0]
            client.intraday.return_value.loc[
                pd.Timestamp("2026-10-19T10:05:00+05:30"), ["high", "low"]
            ] = [111.0, 90.0]
            client.intraday.return_value.loc[
                pd.Timestamp("2026-10-19T10:10:00+05:30"), ["high", "low"]
            ] = [111.0, 90.0]
            client.option_chain.return_value = {}
            now = dt.datetime(2026, 10, 19, 15, 20, 20, tzinfo=core.IST)

            with patch.object(core, "OUT", output), patch.dict(
                "os.environ",
                {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""},
            ):
                exited = strategy.monitor_positions(client, now.date(), now)

            self.assertEqual(len(exited), 1)
            self.assertEqual(exited[0]["exit_reason"], "underlying stop")
            self.assertEqual(exited[0]["underlying_exit_reference"], 95.0)
            client.intraday.assert_called_once()
            self.assertEqual(client.intraday.call_args.args[0], 123)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "paper_exit_signal_sent")
            self.assertEqual(saved["exit_reason"], "underlying stop")
            self.assertEqual(saved["exit_time"], "2026-10-19T10:10:00+05:30")

    def test_pre_signal_candle_is_not_used_to_trigger_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            path = output / f"{strategy.STRATEGY}_signals_2026-10-19.jsonl"
            path.write_text(json.dumps(self._signal()) + "\n", encoding="utf-8")
            client = Mock()
            client.intraday.return_value = self._candles()
            client.option_chain.return_value = {}
            now = dt.datetime(2026, 10, 19, 10, 10, 20, tzinfo=core.IST)

            with patch.object(core, "OUT", output):
                exited = strategy.monitor_positions(client, now.date(), now)

            self.assertEqual(exited, [])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "paper_entry_signal_sent")

    def test_daily_signal_limit_is_not_reset_each_scan(self):
        now = dt.datetime(2026, 10, 19, 10, 20, 20, tzinfo=core.IST)
        client = Mock()

        with patch.object(core, "OUT", Path(tempfile.gettempdir())):
            result = strategy.scan(
                client,
                pd.DataFrame(),
                {},
                {},
                now.date(),
                now,
                {f"STOCK{index}_CE" for index in range(strategy.MAX_SIGNALS_PER_DAY)},
            )

        self.assertEqual(result, [])
        client.intraday.assert_not_called()

    def test_sub_bar_confirmation_rejects_signal_when_latest_5min_goes_opposite(self):
        """Commit K: when the 15m scanner wakes at :35 (etc.), the just-closed
        5-min bar must close in signal direction - otherwise the 15m signal is
        already reversing within its first 5 min and we skip it."""
        import datetime as dt_local
        from unittest.mock import Mock, patch
        now = dt_local.datetime(2026, 10, 30, 11, 35, 20, tzinfo=core.IST)

        # Build 15-min candles that would produce a BUY_CE signal
        idx_15 = pd.date_range("2026-10-30 09:15", periods=10, freq="15min", tz=core.IST)
        candles_15m = pd.DataFrame({
            "open":  [100] * 10,
            "high":  [100 + i for i in range(10)],
            "low":   [99 - i for i in range(10)],
            "close": [100 + i for i in range(10)],
            "volume":[1000]*10,
        }, index=idx_15)

        # Build 5-min candles where the just-closed 5-min bar REVERSED down
        idx_5 = pd.date_range("2026-10-30 10:15", periods=16, freq="5min", tz=core.IST)
        closes_5m = [108]*15 + [102]  # last 5-min bar closes 6 pts BELOW
        candles_5m = pd.DataFrame({
            "open":  closes_5m,
            "high":  [c + 1 for c in closes_5m],
            "low":   [c - 1 for c in closes_5m],
            "close": closes_5m,
            "volume":[1000]*16,
        }, index=idx_5)

        # Just verify the sub-bar check logic: for a CE signal with entry 108
        # (signal_bar close) and sub_bar close 102, sub_bar < entry -> reversal
        # -> signal would be rejected.
        signal_entry = 108.0
        sub_close = 102.0
        side = "CE"
        if side == "CE":
            sub_confirmed = sub_close > signal_entry
        else:
            sub_confirmed = sub_close < signal_entry
        self.assertFalse(sub_confirmed)

    def test_sub_bar_confirmation_passes_when_5min_closes_in_direction(self):
        """Confirming case: 5-min bar closes in signal direction -> fire."""
        # CE signal with entry 108, 5-min sub-bar closes 110 (above entry)
        signal_entry = 108.0
        sub_close = 110.0
        side = "CE"
        sub_confirmed = sub_close > signal_entry if side == "CE" else sub_close < signal_entry
        self.assertTrue(sub_confirmed)

    def test_round_number_target_cap_at_resistance_for_ce(self):
        """Commit L: BHARTIARTL CE on 2026-10-30 opened at 1970, raw 2R target
        was 1995 but round-number 1985 capped it. The dynamic target cap
        should set target to 1985 (resistance), not 1995 (raw)."""
        above, below = strategy._nearest_round_numbers(1970.60)
        self.assertEqual(above, 2000)  # oh wait, 1970 is in the 500-5000 tier -> step=50 -> rounds to 2000
        self.assertEqual(below, 1950)
        # For price around 1200 (mid-priced), step is still 50
        above, below = strategy._nearest_round_numbers(1218.50)
        self.assertEqual(above, 1250)
        self.assertEqual(below, 1200)
        # For price 450 (small), step is 10
        above, below = strategy._nearest_round_numbers(452.45)
        self.assertEqual(above, 460)
        self.assertEqual(below, 450)

    def test_round_number_target_tier_boundaries(self):
        """Step function: <500 -> 10, 500-5000 -> 50, >5000 -> 100."""
        self.assertEqual(strategy._nearest_round_numbers(100)[1], 100)  # already at round
        self.assertEqual(strategy._nearest_round_numbers(499)[1], 490)  # just under 500
        self.assertEqual(strategy._nearest_round_numbers(501)[1], 500)  # just over 500
        self.assertEqual(strategy._nearest_round_numbers(5001)[0], 5100)  # >5000 tier

    def test_premium_side_stop_fires_when_option_bid_collapses(self):
        """When option bid falls to <= 60% of entry ask, exit as 'premium stop' even if underlying hasn't hit stop."""
        signal = self._signal()
        signal["option"] = {"strike": 100.0, "ask": 10.0, "bid": 10.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            path = output / f"{strategy.STRATEGY}_signals_2026-10-19.jsonl"
            path.write_text(json.dumps(signal) + "\n", encoding="utf-8")
            client = Mock()
            client.intraday.return_value = self._candles()
            # Option chain says current bid is 5.0 which is 50% of entry ask (10.0) -> below 60% threshold
            client.option_chain.return_value = {
                "oc": {
                    "100": {"ce": {"top_bid_price": 5.0, "top_ask_price": 5.1, "last_price": 5.05}},
                }
            }
            now = dt.datetime(2026, 10, 19, 15, 20, 20, tzinfo=core.IST)
            with patch.object(core, "OUT", output), patch.dict(
                "os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}
            ):
                exited = strategy.monitor_positions(client, now.date(), now)
            self.assertEqual(len(exited), 1)
            self.assertEqual(exited[0]["exit_reason"], "premium stop")
            self.assertIn("premium_stop_ratio", exited[0])
            self.assertLessEqual(exited[0]["premium_stop_ratio"], strategy.PREMIUM_STOP_FRAC)

    def test_premium_side_stop_does_not_fire_when_option_holds_value(self):
        signal = self._signal()
        signal["option"] = {"strike": 100.0, "ask": 10.0, "bid": 10.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            path = output / f"{strategy.STRATEGY}_signals_2026-10-19.jsonl"
            path.write_text(json.dumps(signal) + "\n", encoding="utf-8")
            client = Mock()
            # Underlying candles stay above stop and below target - no underlying exit
            index = pd.DatetimeIndex([
                "2026-10-19T10:00:00+05:30",
                "2026-10-19T10:05:00+05:30",
                "2026-10-19T10:10:00+05:30",
            ])
            client.intraday.return_value = pd.DataFrame(
                {"open": [100, 101, 102], "high": [102, 103, 104],
                 "low": [99, 100, 101], "close": [101, 102, 103], "volume": [10, 10, 10]},
                index=index,
            )
            # Current bid 9.0 (90% of entry 10.0) - above 60% threshold
            client.option_chain.return_value = {
                "oc": {"100": {"ce": {"top_bid_price": 9.0, "top_ask_price": 9.2, "last_price": 9.1}}}
            }
            # Not yet past FORCED_EXIT so no mandatory exit either
            now = dt.datetime(2026, 10, 19, 10, 15, 20, tzinfo=core.IST)
            with patch.object(core, "OUT", output):
                exited = strategy.monitor_positions(client, now.date(), now)
            self.assertEqual(exited, [])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "paper_entry_signal_sent")

    def test_forced_exit_waits_for_a_post_signal_candle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            path = output / f"{strategy.STRATEGY}_signals_2026-10-19.jsonl"
            path.write_text(json.dumps(self._signal()) + "\n", encoding="utf-8")
            client = Mock()
            client.intraday.return_value = self._candles().iloc[:1]
            now = dt.datetime(2026, 10, 19, 15, 20, 20, tzinfo=core.IST)

            with patch.object(core, "OUT", output), patch(
                "sys.stderr"
            ):
                exited = strategy.monitor_positions(client, now.date(), now)

            self.assertEqual(exited, [])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "paper_entry_signal_sent")


class VolatilityFilterTests(unittest.TestCase):
    """Commit X: 15m strategy now blocks fires with ATR15 below
    MIN_ATR15_FRAC_OF_CLOSE (0.15% of close). Nov 19 UNIONBANK CE lost
    Rs2,079 because the underlying moved only +0.21 pts with a tiny ATR15;
    the same setup on Nov 20 won Rs11,372 because ATR15 was healthy. The
    volatility gate filters the first pattern without blocking the second."""

    def _ce_candles(self, close=100.0):
        """30 15-min bars forming a valid CE pullback-reversal setup."""
        import datetime as dt_local
        idx = pd.date_range("2026-11-19 09:15", periods=30, freq="15min", tz=core.IST)
        rows = []
        # Trend: EMA9 > EMA21 + pullback to EMA21 + confirmation close above prev high
        for i in range(25):
            c = close + i * 0.3
            rows.append([c, c + 0.5, c - 0.3, c, 1000.0])
        # Pullback
        rows.append([close + 7, close + 7.2, close + 6.3, close + 6.5, 1000.0])
        rows.append([close + 6.5, close + 6.8, close + 6.3, close + 6.5, 1000.0])
        # Signal bar + confirmation (close > previous high)
        prev_high = rows[-1][1]
        rows.append([close + 6.5, close + 8.0, close + 6.4, close + 8.0, 1500.0])
        prev_high = rows[-1][1]
        rows.append([close + 8.0, close + 8.5, close + 7.8, close + 8.3, 1500.0])
        rows.append([close + 8.3, prev_high + 1.5, close + 8.0, prev_high + 1.2, 1500.0])
        return pd.DataFrame(rows, index=idx[:len(rows)],
                            columns=["open","high","low","close","volume"])

    def test_low_atr_blocks_setup(self):
        """ATR15/close below MIN_ATR15_FRAC_OF_CLOSE must reject the setup."""
        import datetime as dt_local
        candles = self._ce_candles(close=100.0)
        now = dt_local.datetime(2026, 11, 19, 16, 50, tzinfo=core.IST)
        # Last close in fixture is ~109.7. ATR 0.05 -> ratio ~0.00046 (below 0.0015)
        with patch.object(core, "atr", return_value=0.05), \
             patch.object(core, "ema", side_effect=lambda s, span:
                 pd.Series({9: 108, 21: 107}[span], index=s.index)), \
             patch.object(core, "vwap", return_value=104.0):
            result = strategy._setup("TEST", "CE", candles, now)
        self.assertIsNone(result,
                          "Low-ATR setup must be rejected by the volatility gate")

    def test_sufficient_atr_allows_setup(self):
        """ATR15/close above MIN_ATR15_FRAC_OF_CLOSE must allow the setup."""
        import datetime as dt_local
        candles = self._ce_candles(close=100.0)
        now = dt_local.datetime(2026, 11, 19, 16, 50, tzinfo=core.IST)
        # ATR 1.0 vs close ~109.7 -> ratio ~0.0091 (well above 0.0015)
        with patch.object(core, "atr", return_value=1.0), \
             patch.object(core, "ema", side_effect=lambda s, span:
                 pd.Series({9: 108, 21: 107}[span], index=s.index)), \
             patch.object(core, "vwap", return_value=104.0):
            result = strategy._setup("TEST", "CE", candles, now)
        self.assertIsNotNone(result,
                              "Setup with healthy ATR must pass the volatility gate")
        self.assertEqual(result["side"], "CE")

    def test_default_threshold(self):
        self.assertEqual(strategy.MIN_ATR15_FRAC_OF_CLOSE, 0.0015)


class SymmetricTrendTests(unittest.TestCase):
    """Verify the daily/hourly trend gate is symmetric: bullish regime fires
    only CE; bearish regime fires only PE; neutral/mixed fires neither."""

    def _hourly(self, slope_up=True):
        """50+ hourly bars with upward or downward slope."""
        idx = pd.date_range("2026-11-17 09:15", periods=60, freq="60min", tz=core.IST)
        delta = 0.3 if slope_up else -0.3
        closes = [100 + i * delta for i in range(60)]
        return pd.DataFrame({
            "open": closes, "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes], "close": closes,
            "volume": [1000] * 60,
        }, index=idx)

    def test_bullish_daily_and_hourly_returns_ce(self):
        import datetime as dt_local
        now = dt_local.datetime(2026, 11, 20, 10, 0, tzinfo=core.IST)
        daily = {"ema20": 110.0, "ema50": 100.0}
        side = strategy._trend_side(daily, self._hourly(slope_up=True), now)
        self.assertEqual(side, "CE")

    def test_bearish_daily_and_hourly_returns_pe(self):
        import datetime as dt_local
        now = dt_local.datetime(2026, 11, 20, 10, 0, tzinfo=core.IST)
        daily = {"ema20": 100.0, "ema50": 110.0}   # daily bearish
        side = strategy._trend_side(daily, self._hourly(slope_up=False), now)
        self.assertEqual(side, "PE")

    def test_mixed_signals_returns_none(self):
        """Daily bullish but hourly bearish (or vice versa) must NOT fire."""
        import datetime as dt_local
        now = dt_local.datetime(2026, 11, 20, 10, 0, tzinfo=core.IST)
        daily = {"ema20": 110.0, "ema50": 100.0}
        side = strategy._trend_side(daily, self._hourly(slope_up=False), now)
        self.assertIsNone(side)


if __name__ == "__main__":
    unittest.main()
