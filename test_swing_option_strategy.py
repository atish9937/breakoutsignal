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


class SwingPostTriggerOutcomeTests(unittest.TestCase):
    """Commit V: once a swing candidate's trigger is confirmed, monitor_confirmed_outcomes()
    must classify what actually happened to the underlying during the hold window
    (max 3 sessions OR Friday close, whichever is first)."""

    def _confirmed_candidate(self, side="CE", confirmed_at="2026-11-02T10:45:00+05:30"):
        # Monday 2026-11-02; hold-end should be min(Wed 2026-11-04, Fri 2026-11-06) = Wed.
        return {
            "symbol": "TEST",
            "side": side,
            "setup": "Daily pullback and reversal",
            "signal_date": "2026-10-30",
            "score": 5,
            "nifty_bias": "UP" if side == "CE" else "DOWN",
            "underlying": {
                "close": 100.0,
                "entry_trigger": 101.0,
                "stop": 98.0,
                "target_2r": 107.0,
                "risk": 3.0,
                "atr14": 2.0,
            },
            "holding": {"max_trading_sessions": 3, "weekend_carry": False},
            "underlying_security_id": 1234,
            "expiry": "2026-11-27",
            "next_session_status": "trigger_confirmed",
            "confirmed_at": confirmed_at,
            "confirmed_underlying_close": 101.5,
            "refreshed_option": {"strike": 100, "selection": "ATM",
                                  "bid": 5.0, "ask": 5.2, "spread_pct": 3.9},
        }

    def _write_report(self, root: Path, day: str, candidate: dict) -> Path:
        import json as _json
        path = root / f"swing_option_candidates_{day}.json"
        path.write_text(_json.dumps({
            "as_of": day, "candidates": [candidate],
        }), encoding="utf-8")
        return path

    def _fake_dh(self, daily_df=None, intraday_df=None):
        class _Dh:
            def daily_range(self, sid, frm, to):
                return daily_df if daily_df is not None else pd.DataFrame()
            def intraday(self, sid, frm, to, interval=5, **kw):
                return intraday_df if intraday_df is not None else pd.DataFrame()
        return _Dh()

    def test_target_hit_resolves_as_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cand = self._confirmed_candidate(side="CE")
            self._write_report(root, "2026-10-30", cand)
            # Day after confirmation: high reaches 108 (>= target 107), low 100 (> stop 98)
            dates = pd.DatetimeIndex([pd.Timestamp("2026-11-03", tz=IST)])
            daily = pd.DataFrame({"open":[102.0], "high":[108.0], "low":[100.0],
                                   "close":[107.5], "volume":[100.0]}, index=dates)
            resolved = strategy.monitor_confirmed_outcomes(
                self._fake_dh(daily_df=daily),
                None,
                dt.date(2026, 11, 3),
                dt.datetime(2026, 11, 3, 15, 35, tzinfo=IST),
                root,
            )
            self.assertEqual(len(resolved), 1)
            outcome = resolved[0]["next_session_outcome"]
            self.assertEqual(outcome["status"], "target_hit")
            self.assertEqual(outcome["pnl_r"], 2.0)

    def test_stop_hit_resolves_as_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cand = self._confirmed_candidate(side="CE")
            self._write_report(root, "2026-10-30", cand)
            # Day after confirmation: low 97 (< stop 98), high 103 (< target 107)
            dates = pd.DatetimeIndex([pd.Timestamp("2026-11-03", tz=IST)])
            daily = pd.DataFrame({"open":[101.0], "high":[103.0], "low":[97.0],
                                   "close":[99.0], "volume":[100.0]}, index=dates)
            resolved = strategy.monitor_confirmed_outcomes(
                self._fake_dh(daily_df=daily), None,
                dt.date(2026, 11, 3),
                dt.datetime(2026, 11, 3, 15, 35, tzinfo=IST),
                root,
            )
            outcome = resolved[0]["next_session_outcome"]
            self.assertEqual(outcome["status"], "stop_hit")
            self.assertEqual(outcome["pnl_r"], -1.0)

    def test_forced_exit_at_hold_end(self):
        """Neither target nor stop hit; hold window expires -> forced exit at
        last close with realised R multiple."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cand = self._confirmed_candidate(side="CE")
            self._write_report(root, "2026-10-30", cand)
            # Hold window = Mon 11-02 (confirmed), Tue 11-03, Wed 11-04 (hold_end)
            # All days price drifts between stop and target.
            dates = pd.DatetimeIndex([
                pd.Timestamp("2026-11-02", tz=IST),
                pd.Timestamp("2026-11-03", tz=IST),
                pd.Timestamp("2026-11-04", tz=IST),
            ])
            daily = pd.DataFrame({
                "open":  [101.0, 102.0, 103.0],
                "high":  [102.0, 104.0, 104.5],
                "low":   [100.0, 100.5, 101.0],
                "close": [102.0, 103.5, 104.0],
                "volume":[100.0]*3,
            }, index=dates)
            resolved = strategy.monitor_confirmed_outcomes(
                self._fake_dh(daily_df=daily), None,
                dt.date(2026, 11, 4),
                dt.datetime(2026, 11, 4, 15, 35, tzinfo=IST),
                root,
            )
            outcome = resolved[0]["next_session_outcome"]
            self.assertEqual(outcome["status"], "forced_exit")
            # Close 104 vs entry 101 -> +3 pts on R=3 -> +1.0R
            self.assertEqual(outcome["pnl_r"], 1.0)

    def test_pe_side_mirrors_direction(self):
        """PE candidate: hit when price falls below target_2r."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cand = self._confirmed_candidate(side="PE")
            # PE: entry 101, stop 98 (above for PE must be above entry);
            # need PE-style levels: stop > entry > target. Rebuild manually:
            cand["underlying"] = {"close":100, "entry_trigger":99, "stop":102,
                                   "target_2r":93, "risk":3, "atr14":2}
            self._write_report(root, "2026-10-30", cand)
            dates = pd.DatetimeIndex([pd.Timestamp("2026-11-03", tz=IST)])
            daily = pd.DataFrame({"open":[98.0], "high":[99.0], "low":[92.0],
                                   "close":[93.5], "volume":[100.0]}, index=dates)
            resolved = strategy.monitor_confirmed_outcomes(
                self._fake_dh(daily_df=daily), None,
                dt.date(2026, 11, 3),
                dt.datetime(2026, 11, 3, 15, 35, tzinfo=IST),
                root,
            )
            outcome = resolved[0]["next_session_outcome"]
            self.assertEqual(outcome["status"], "target_hit")

    def test_resolver_is_idempotent(self):
        """Running twice on a candidate with next_session_outcome already set
        must not duplicate outcomes or re-call Dhan."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cand = self._confirmed_candidate(side="CE")
            self._write_report(root, "2026-10-30", cand)
            dates = pd.DatetimeIndex([pd.Timestamp("2026-11-03", tz=IST)])
            daily = pd.DataFrame({"open":[102.0], "high":[108.0], "low":[100.0],
                                   "close":[107.5], "volume":[100.0]}, index=dates)
            dh = self._fake_dh(daily_df=daily)
            r1 = strategy.monitor_confirmed_outcomes(
                dh, None, dt.date(2026, 11, 3),
                dt.datetime(2026, 11, 3, 15, 35, tzinfo=IST), root,
            )
            r2 = strategy.monitor_confirmed_outcomes(
                dh, None, dt.date(2026, 11, 3),
                dt.datetime(2026, 11, 3, 15, 35, tzinfo=IST), root,
            )
            self.assertEqual(len(r1), 1)
            self.assertEqual(len(r2), 0)

    def test_non_confirmed_candidates_are_ignored(self):
        """Resolver must not touch rows with status != 'trigger_confirmed'."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cand = self._confirmed_candidate(side="CE")
            cand["next_session_status"] = "pending"
            self._write_report(root, "2026-10-30", cand)
            dates = pd.DatetimeIndex([pd.Timestamp("2026-11-03", tz=IST)])
            daily = pd.DataFrame({"open":[102.0], "high":[108.0], "low":[100.0],
                                   "close":[107.5], "volume":[100.0]}, index=dates)
            resolved = strategy.monitor_confirmed_outcomes(
                self._fake_dh(daily_df=daily), None,
                dt.date(2026, 11, 3),
                dt.datetime(2026, 11, 3, 15, 35, tzinfo=IST),
                root,
            )
            self.assertEqual(resolved, [])

    def test_hold_end_date_respects_friday_cutoff(self):
        """Confirmed Thursday 2026-11-05: 3 sessions = Thu+Fri+Mon (Mon 2026-11-09).
        But Friday-close cutoff clamps to Fri 2026-11-06."""
        from swing_option_strategy import _hold_end_date
        self.assertEqual(_hold_end_date(dt.date(2026, 11, 5)),
                         dt.date(2026, 11, 6))
        # Confirmed Monday 2026-11-02: 3 sessions = Mon+Tue+Wed (2026-11-04).
        # Friday is 2026-11-06, so 3-session rule is tighter.
        self.assertEqual(_hold_end_date(dt.date(2026, 11, 2)),
                         dt.date(2026, 11, 4))


if __name__ == "__main__":
    unittest.main()
