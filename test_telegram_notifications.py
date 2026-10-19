import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import breakout_scanner as scanner


class TelegramNotificationTests(unittest.TestCase):
    def setUp(self):
        self.signal = {
            "verdict": "BUY_CE",
            "symbol": "TEST",
            "ltp": 100.0,
            "trigger_level": 99.5,
            "option": {"contract": "TEST CE", "premium": 10.0, "lot_size": 50},
            "plan": {"stop": 98.0, "t1": 102.0, "t2": 104.0},
        }
        self.environment = patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "bot-secret", "TELEGRAM_CHAT_ID": "123"},
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_load_env_reads_file_and_keeps_exported_environment_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "DHAN_CLIENT_ID=file-client\n"
                "DHAN_ACCESS_TOKEN='file-token-value'\n"
                "TELEGRAM_CHAT_ID=@signal_channel\n"
                "ENABLE_LIVE_ORDERS=true\n",
                encoding="utf-8",
            )
            with patch.object(scanner, "ENV_FILE", env_file):
                with patch.dict(os.environ, {"DHAN_CLIENT_ID": "exported-client"}, clear=True):
                    scanner.load_env()

                    self.assertEqual(os.environ["DHAN_CLIENT_ID"], "exported-client")
                    self.assertEqual(os.environ["DHAN_ACCESS_TOKEN"], "file-token-value")
                    self.assertEqual(os.environ["TELEGRAM_CHAT_ID"], "@signal_channel")
                    self.assertEqual(os.environ["ENABLE_LIVE_ORDERS"], "true")

    def test_credentials_are_loaded_from_env_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("DHAN_CLIENT_ID=client\nDHAN_ACCESS_TOKEN=token\n", encoding="utf-8")
            with patch.object(scanner, "ENV_FILE", env_file):
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(scanner.credentials(), ("client", "token"))

    @patch.object(scanner.requests, "post")
    def test_sends_new_signal_to_telegram(self, post):
        response = Mock()
        response.json.return_value = {"ok": True}
        post.return_value = response

        scanner.notify(self.signal)

        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.telegram.org/botbot-secret/sendMessage")
        self.assertEqual(kwargs["json"]["chat_id"], "123")
        self.assertIn("BUY_CE TEST", kwargs["json"]["text"])
        self.assertEqual(kwargs["timeout"], 10)
        response.raise_for_status.assert_called_once()

    @patch.object(scanner.requests, "post")
    def test_reports_request_failure_without_leaking_token(self, post):
        post.side_effect = requests.ConnectionError("network unavailable")

        with patch("sys.stderr") as stderr:
            scanner.notify(self.signal)

        self.assertIn("Telegram notification failed", "".join(
            str(call.args[0]) for call in stderr.write.call_args_list
        ))
        self.assertNotIn("bot-secret", "".join(
            str(call.args[0]) for call in stderr.write.call_args_list
        ))

    @patch.object(scanner.requests, "post")
    def test_reports_telegram_api_error(self, post):
        response = Mock()
        response.json.return_value = {"ok": False, "description": "chat not found"}
        post.return_value = response

        with patch("sys.stderr") as stderr:
            scanner.notify(self.signal)

        self.assertIn("chat not found", "".join(
            str(call.args[0]) for call in stderr.write.call_args_list
        ))

    @patch.object(scanner.requests, "post")
    def test_sends_completion_message_when_scan_has_no_buy_signals(self, post):
        response = Mock()
        response.json.return_value = {"ok": True}
        post.return_value = response

        scanner.notify_scan_completion({
            "scan_time": "2026-10-19 10:00:00",
            "candidates_checked": 2,
            "results": [
                {"verdict": "WATCH_CE", "symbol": "AAA"},
                {"verdict": "SKIP_PE", "symbol": "BBB"},
            ],
            "nifty": {"ltp": 25000, "chg_pct": 0.5, "bias": "BULLISH"},
        })

        post.assert_called_once()
        message = post.call_args.kwargs["json"]["text"]
        self.assertIn("Scan complete - 2026-10-19 10:00:00 IST", message)
        self.assertIn("Candidates: 2", message)
        self.assertIn("BUY: 0 | WATCH: 1 | SKIP: 1", message)
        self.assertIn("No BUY signals this scan.", message)

    @patch.object(scanner.requests, "post")
    def test_completion_message_lists_buy_signals(self, post):
        response = Mock()
        response.json.return_value = {"ok": True}
        post.return_value = response

        scanner.notify_scan_completion({
            "scan_time": "2026-10-19 10:05:00",
            "candidates_checked": 1,
            "results": [{"verdict": "BUY_PE", "symbol": "AAA", "ltp": 123.4}],
            "nifty": {"ltp": 25000, "chg_pct": -0.2, "bias": "BEARISH"},
        })

        message = post.call_args.kwargs["json"]["text"]
        self.assertIn("BUY: 1 | WATCH: 0 | SKIP: 0", message)
        self.assertIn("BUY_PE AAA @ 123.4", message)


if __name__ == "__main__":
    unittest.main()
