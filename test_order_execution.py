import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

import breakout_scanner as scanner


class OrderPayloadTests(unittest.TestCase):
    def signal(self):
        return {
            "verdict": "BUY_CE",
            "symbol": "TEST",
            "side": "CE",
            "option": {
                "type": "CE",
                "security_id": 12345,
                "lot_size": 25,
                "ask": 12.35,
            },
        }

    def test_builds_one_lot_intraday_limit_buy(self):
        payload = scanner.build_order_payload(
            self.signal(), "1234567890", dt.date(2026, 10, 19)
        )

        self.assertEqual(payload["dhanClientId"], "1234567890")
        self.assertEqual(payload["transactionType"], "BUY")
        self.assertEqual(payload["exchangeSegment"], "NSE_FNO")
        self.assertEqual(payload["productType"], "INTRADAY")
        self.assertEqual(payload["orderType"], "LIMIT")
        self.assertEqual(payload["securityId"], "12345")
        self.assertEqual(payload["quantity"], 25)
        self.assertEqual(payload["price"], 12.35)
        self.assertEqual(len(payload["correlationId"]), 18)

    def test_refuses_missing_exact_scrip_id(self):
        signal = self.signal()
        signal["option"]["security_id"] = None

        with self.assertRaisesRegex(ValueError, "security ID"):
            scanner.build_order_payload(signal, "1234567890", dt.date.today())

    def test_refuses_risk_warning(self):
        signal = self.signal()
        signal["risk_warning"] = "over limit"

        with self.assertRaisesRegex(ValueError, "risk limit"):
            scanner.build_order_payload(signal, "1234567890", dt.date.today())


class DhanOrderSubmissionTests(unittest.TestCase):
    def make_client(self):
        client = object.__new__(scanner.Dhan)
        client.s = Mock()
        return client

    def test_place_order_uses_single_non_retried_post(self):
        client = self.make_client()
        response = Mock()
        response.json.return_value = {"orderId": "987", "orderStatus": "PENDING"}
        client.s.post.return_value = response

        result = client.place_order({"orderType": "LIMIT"})

        self.assertEqual(result["orderId"], "987")
        client.s.post.assert_called_once_with(
            scanner.API + "/orders",
            json={"orderType": "LIMIT"},
            timeout=(10, 30),
        )
        response.raise_for_status.assert_called_once()

    def test_place_order_does_not_retry_timeout(self):
        client = self.make_client()
        client.s.post.side_effect = requests.Timeout("response lost")

        with self.assertRaises(requests.Timeout):
            client.place_order({})

        client.s.post.assert_called_once()

    def test_live_submission_records_order_id_and_status(self):
        client = self.make_client()
        client.client_id = "1234567890"
        client.place_order = Mock(return_value={"orderId": "987", "orderStatus": "PENDING"})
        signal = OrderPayloadTests().signal()

        with tempfile.TemporaryDirectory() as temp_dir:
            with unittest.mock.patch.object(scanner, "OUT", Path(temp_dir)):
                result = scanner.submit_live_order(client, signal, dt.date(2026, 10, 19))
                events = [
                    json.loads(line)
                    for line in (Path(temp_dir) / "orders_2026-10-19.jsonl").read_text().splitlines()
                ]

        self.assertEqual(result["status"], "SUBMITTED")
        self.assertEqual(result["order_id"], "987")
        self.assertEqual([event["status"] for event in events], ["SUBMITTING", "SUBMITTED"])


if __name__ == "__main__":
    unittest.main()
