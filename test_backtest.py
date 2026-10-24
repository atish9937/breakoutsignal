import datetime as dt
import threading
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

import backtest
import breakout_scanner as scanner


class BacktestExecutionTests(unittest.TestCase):
    def make_signal(self):
        return {
            "symbol": "TEST",
            "side": "CE",
            "date": "2025-01-02",
            "signal_time": "10:05:20",
            "signal_index": 0,
            "level": 100.0,
            "atr": 28.5714285714,
            "quality": 4,
            "checks": {},
        }

    def make_bars(self, rows):
        index = pd.date_range("2025-01-02 10:05", periods=len(rows), freq="5min",
                              tz=backtest.IST)
        return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])

    def test_stop_wins_when_stop_and_target_are_both_hit(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 1000],
            [100, 130, 89, 100, 1000],
        ])
        trade = backtest._execute_trade(self.make_signal(), bars, 0, 0)
        self.assertEqual(trade["exit_reason"], "stop")
        self.assertAlmostEqual(trade["r_multiple"], -1.0, places=4)

    def test_half_at_t1_then_breakeven_on_same_bar(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 1000],
            [100, 116, 100, 105, 1000],
        ])
        trade = backtest._execute_trade(self.make_signal(), bars, 0, 0)
        self.assertEqual(trade["exit_reason"], "t1_then_breakeven_same_bar")
        self.assertAlmostEqual(trade["r_multiple"], 0.75, places=4)

    def test_half_at_t1_and_remaining_half_at_t2(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 1000],
            [100, 126, 101, 125, 1000],
        ])
        trade = backtest._execute_trade(self.make_signal(), bars, 0, 0)
        self.assertEqual(trade["exit_reason"], "t2")
        self.assertAlmostEqual(trade["r_multiple"], 2.0, places=4)

    def test_slippage_and_costs_reduce_net_result(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 1000],
            [100, 126, 101, 125, 1000],
        ])
        free = backtest._execute_trade(self.make_signal(), bars, 0, 0)
        charged = backtest._execute_trade(self.make_signal(), bars, 10, 5)
        self.assertLess(charged["r_multiple"], free["r_multiple"])

    def test_fixed_strike_option_pnl_uses_entry_open_and_exit_close(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 1000],
            [100, 126, 101, 125, 1000],
        ])
        signal = {**self.make_signal(), "option": {"available": True, "strike": 100.0}}
        trade = backtest._execute_trade(signal, bars, 0, 0)
        option_bars = pd.DataFrame(
            [[10, 16, 9, 15, 1000, 500, 20, 100, 100]],
            index=pd.DatetimeIndex([bars.index[1]]),
            columns=["open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot"],
        )
        options = {"CALL_ATM": option_bars}
        result = backtest._option_pnl(signal, trade, options, 0, 0)
        self.assertTrue(result["available"])
        self.assertEqual(result["pnl_per_option_unit"], 5.0)
        self.assertEqual(result["return_pct"], 50.0)

    def test_disabling_chase_filter_allows_next_bar_gap(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 1000],
            [101, 126, 101, 125, 1000],
        ])
        self.assertIsNone(backtest._execute_trade(self.make_signal(), bars, 0, 0))
        trade = backtest._execute_trade(self.make_signal(), bars, 0, 0, disable_chase=True)
        self.assertIsNotNone(trade)

    def test_pe_trade_uses_short_direction_for_targets_and_stop(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 1000],
            [100, 100, 95, 98, 1000],
            [98, 99, 74, 75, 1000],
        ])
        signal = {**self.make_signal(), "side": "PE"}
        trade = backtest._execute_trade(signal, bars, 0, 0)
        self.assertEqual(trade["exit_reason"], "t2")
        self.assertAlmostEqual(trade["r_multiple"], 2.0, places=4)


class ExitVariantTests(unittest.TestCase):
    make_signal = BacktestExecutionTests.make_signal
    make_bars = BacktestExecutionTests.make_bars
    def test_empty_rules_match_default_plan(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 1000],
            [100, 126, 101, 125, 1000],
        ])
        base = backtest._execute_trade(self.make_signal(), bars, 0, 0)
        same = backtest._execute_trade(self.make_signal(), bars, 0, 0,
                                       exit_rules=backtest.EXIT_VARIANTS["live_plan"])
        self.assertEqual(base["r_multiple"], same["r_multiple"])
        self.assertEqual(base["exit_reason"], same["exit_reason"])

    def test_full_exit_at_t1(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 1000],
            [100, 116, 100, 105, 1000],
        ])
        trade = backtest._execute_trade(self.make_signal(), bars, 0, 0,
                                        exit_rules=backtest.EXIT_VARIANTS["full_exit_at_t1"])
        self.assertEqual(trade["exit_reason"], "t1")
        self.assertAlmostEqual(trade["r_multiple"], 1.5, places=4)

    def test_breakeven_after_one_r_applies_from_next_bar(self):
        bars = self.make_bars([
            [100, 101, 99, 100, 1000],
            [100, 111, 89, 100, 1000],
            [100, 101, 99, 100, 1000],
        ])
        trade = backtest._execute_trade(self.make_signal(), bars, 0, 0,
                                        exit_rules=backtest.EXIT_VARIANTS["be_after_1r"])
        self.assertAlmostEqual(trade["r_multiple"], -1.0, places=4)
        bars = self.make_bars([
            [100, 101, 99, 100, 1000],
            [100, 111, 100, 105, 1000],
            [105, 106, 99, 100, 1000],
        ])
        trade = backtest._execute_trade(self.make_signal(), bars, 0, 0,
                                        exit_rules=backtest.EXIT_VARIANTS["be_after_1r"])
        self.assertAlmostEqual(trade["r_multiple"], 0.0, places=4)


class DhanRetryTests(unittest.TestCase):
    def test_rolling_option_retries_read_timeout(self):
        client = object.__new__(scanner.Dhan)
        client.s = Mock()
        client.s.post.side_effect = [
            requests.Timeout("read timed out"),
            Mock(status_code=200, json=Mock(return_value={"data": {}})),
        ]
        client._last = {}
        client._last_request = None
        client._request_lock = threading.Lock()
        with patch.object(scanner.time, "sleep"):
            result = client.rolling_option({"strike": "ATM"})
        self.assertEqual(result, {"data": {}})
        self.assertEqual(client.s.post.call_count, 2)
        self.assertEqual(client.s.post.call_args.kwargs["timeout"], (10, 90))

    def test_data_request_gap_is_shared_across_endpoints(self):
        client = scanner.Dhan("client", "token")
        response = Mock(status_code=200, json=Mock(return_value={}))
        client.s.post = Mock(return_value=response)
        current_time = [100.0]
        waits = []

        def fake_sleep(seconds):
            waits.append(seconds)
            current_time[0] += seconds

        with patch.object(scanner.time, "monotonic", side_effect=lambda: current_time[0]):
            with patch.object(scanner.time, "sleep", side_effect=fake_sleep):
                client._post("/charts/intraday", {}, min_gap=0)
                client._post("/marketfeed/ohlc", {}, min_gap=0)

        self.assertEqual(len(waits), 1)
        self.assertAlmostEqual(waits[0], scanner.DHAN_DATA_MIN_REQUEST_GAP_SECONDS)
        self.assertEqual(client.s.post.call_count, 2)

    def test_option_chain_rate_limit_is_shared_with_expiry_list(self):
        client = scanner.Dhan("client", "token")
        response = Mock(status_code=200, json=Mock(return_value={"data": []}))
        client.s.post = Mock(return_value=response)
        current_time = [100.0]
        waits = []

        def fake_sleep(seconds):
            waits.append(seconds)
            current_time[0] += seconds

        with patch.object(scanner.time, "monotonic", side_effect=lambda: current_time[0]):
            with patch.object(scanner.time, "sleep", side_effect=fake_sleep):
                client.expiries(123)
                client.option_chain(123, "2026-10-29")

        self.assertEqual(len(waits), 1)
        self.assertAlmostEqual(waits[0], 3.1)
        self.assertEqual(client.s.post.call_count, 2)


class BacktestMetricsTests(unittest.TestCase):
    def test_metrics_include_total_r_and_drawdown(self):
        trades = [{"r_multiple": 1.0}, {"r_multiple": -0.5}, {"r_multiple": 2.0}]
        metrics = backtest._metrics(trades)
        self.assertEqual(metrics["trades"], 3)
        self.assertEqual(metrics["wins"], 2)
        self.assertEqual(metrics["total_r"], 2.5)
        self.assertEqual(metrics["max_drawdown_r"], 0.5)


class BacktestSignalTests(unittest.TestCase):
    def make_option_frames(self, timestamps):
        fields = ["open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot"]

        def frame(strike, premium, oi):
            return pd.DataFrame(
                [[premium, premium + 0.5, premium - 0.5, premium, 1000, oi, 20, strike, 100.0]
                 for _ in timestamps],
                index=timestamps, columns=fields,
            )

        return {
            "CALL_ATM": frame(100, 5, 500),
            "CALL_ATM+1": frame(105, 4, 1000),
            "CALL_ATM+2": frame(110, 3, 1200),
            "CALL_ATM+3": frame(115, 2, 300),
            "PUT_ATM": frame(100, 5, 500),
            "PUT_ATM-1": frame(95, 4, 1000),
            "PUT_ATM-2": frame(90, 3, 1200),
            "PUT_ATM-3": frame(85, 2, 300),
        }

    def test_option_snapshot_reconstructs_wall_and_liquidity(self):
        timestamp = pd.Timestamp("2025-01-02 10:00", tz=backtest.IST)
        options = self.make_option_frames(pd.DatetimeIndex([timestamp]))
        snapshot = backtest._option_snapshot(options, "CE", timestamp, 100.0, 10.0, 100.0)
        self.assertTrue(snapshot["option_liquid"])
        self.assertEqual(snapshot["wall"], 105.0)
        self.assertAlmostEqual(snapshot["runway_r"], 1.42857, places=4)

    def test_option_strikes_are_scaled_to_adjusted_underlying_history(self):
        timestamp = pd.Timestamp("2025-01-02 10:00", tz=backtest.IST)
        options = self.make_option_frames(pd.DatetimeIndex([timestamp]))
        for frame in options.values():
            frame["spot"] = 200.0
            frame["strike"] *= 2
        snapshot = backtest._option_snapshot(options, "CE", timestamp, 100.0, 10.0, 100.0)
        self.assertEqual(snapshot["strike"], 200.0)
        self.assertEqual(snapshot["wall"], 105.0)
        self.assertEqual(snapshot["underlying_scale"], 0.5)

    def test_rolling_option_response_is_normalized_to_ist(self):
        timestamps = [1735808400, 1735808700]
        response = {
            "data": {
                "ce": {
                    "timestamp": timestamps,
                    "open": [4.0, 4.1], "high": [4.2, 4.3],
                    "low": [3.9, 4.0], "close": [4.1, 4.2],
                    "volume": [100, 200], "oi": [1000, 1100],
                    "iv": [20, 21], "strike": [100, 100], "spot": [99, 99.5],
                },
                "pe": None,
            }
        }
        frame = backtest._rolling_option_frame(response, "CALL")
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.index[0].utcoffset(), dt.timedelta(hours=5, minutes=30))
        self.assertEqual(frame["oi"].iloc[-1], 1100.0)

    def test_disabling_fresh_breakout_admits_a_stale_break(self):
        index = pd.date_range("2025-01-02 09:45", periods=4, freq="5min",
                              tz=backtest.IST)
        bars = pd.DataFrame([
            [100.1, 100.2, 100.0, 100.1, 100],
            [100.1, 100.3, 100.0, 100.2, 100],
            [100.2, 100.4, 100.1, 100.3, 100],
            [100.3, 100.5, 100.2, 100.4, 150],
        ], index=index, columns=["open", "high", "low", "close", "volume"])
        setup = {
            "atr": 4.0,
            "box_high": 100.0,
            "box_low": 96.0,
            "res_levels": [100.0],
            "sup_levels": [96.0],
            "sma20": 98.0,
        }
        warm = pd.Series([99.0] * 30)
        decision_time = index[-1] + dt.timedelta(minutes=5, seconds=20)
        options = self.make_option_frames(index)
        baseline = backtest._signal_record(
            "TEST", "CE", index[-1].date(), decision_time, bars, 3, setup, warm, "UP", set(), options
        )
        without_fresh = backtest._signal_record(
            "TEST", "CE", index[-1].date(), decision_time, bars, 3, setup, warm, "UP",
            {"fresh_breakout"}, options,
        )
        self.assertIsNone(baseline)
        self.assertIsNotNone(without_fresh)
        self.assertFalse(without_fresh["checks"]["fresh_breakout"])
        self.assertEqual(
            without_fresh["live_price_proxy"],
            "completed 5-minute candle close",
        )

    def test_daily_setup_does_not_use_the_simulated_session(self):
        dates = pd.date_range("2025-01-01", periods=22, freq="D", tz=backtest.IST)
        daily = pd.DataFrame({
            "open": [100.0] * 21 + [500.0],
            "high": [101.0] * 21 + [1000.0],
            "low": [99.0] * 21 + [1.0],
            "close": [100.0] * 21 + [900.0],
            "volume": [1000.0] * 22,
        }, index=dates)
        setup = backtest._daily_setup(daily, dates[-1].date())
        self.assertEqual(setup["box_high"], 101.0)
        self.assertEqual(setup["sma20"], 100.0)

    def test_scenario_comparison_admits_trade_when_volume_gate_is_disabled(self):
        day = dt.date(2025, 1, 22)
        daily_index = pd.date_range("2025-01-01", periods=21, freq="D", tz=backtest.IST)
        daily = pd.DataFrame({
            "open": [98.0] * 21, "high": [100.0] * 21, "low": [96.0] * 21,
            "close": [98.0] * 21, "volume": [1000.0] * 21,
        }, index=daily_index)
        warm_index = pd.date_range(
            pd.Timestamp(day - dt.timedelta(days=1), tz=backtest.IST).replace(hour=9, minute=15),
            periods=30, freq="5min",
        )
        session_index = pd.date_range(
            pd.Timestamp(day, tz=backtest.IST).replace(hour=9, minute=15),
            periods=5, freq="5min",
        )
        stock_warm = pd.DataFrame({
            "open": [99.0] * 30, "high": [99.1] * 30, "low": [98.9] * 30,
            "close": [99.0] * 30, "volume": [100.0] * 30,
        }, index=warm_index)
        stock_session = pd.DataFrame([
            [99.5, 99.7, 99.4, 99.6, 100],
            [99.6, 99.8, 99.5, 99.7, 100],
            [99.7, 100.0, 99.6, 100.0, 100],
            [100.0, 100.3, 99.9, 100.2, 50],
            [100.2, 104.0, 100.2, 104.0, 200],
        ], index=session_index, columns=stock_warm.columns)
        nifty_warm = pd.DataFrame({
            "open": [20000.0 + i for i in range(30)],
            "high": [20001.0 + i for i in range(30)],
            "low": [19999.0 + i for i in range(30)],
            "close": [20000.0 + i for i in range(30)],
            "volume": [1000.0] * 30,
        }, index=warm_index)
        nifty_session = pd.DataFrame({
            "open": [20030.0 + i for i in range(5)],
            "high": [20031.0 + i for i in range(5)],
            "low": [20029.0 + i for i in range(5)],
            "close": [20030.0 + i for i in range(5)],
            "volume": [1000.0] * 5,
        }, index=session_index)
        data = {
            "TEST": {
                "daily": daily, "intraday": pd.concat([stock_warm, stock_session]),
                "options": self.make_option_frames(session_index),
            },
            "NIFTY": {"daily": pd.DataFrame(), "intraday": pd.concat([nifty_warm, nifty_session])},
        }
        baseline = backtest._scenario_trades(data, day, day, set(), 0, 0)
        variant = backtest._scenario_trades(
            data, day, day, {"breakout_volume", "oi_wall_runway", "option_liquidity"}, 0, 0
        )
        self.assertEqual(baseline, [])
        self.assertEqual(len(variant), 1)
        self.assertEqual(variant[0]["symbol"], "TEST")
        self.assertTrue(variant[0]["option_pnl"]["available"])


if __name__ == "__main__":
    unittest.main()
