import datetime as dt
import json
import struct
import tempfile
import unittest
from pathlib import Path

from paper_quote_feed import DhanQuoteRecorder, parse_full_packet
from trade_performance import connect, record_buy_recommendations


class PaperQuoteFeedTests(unittest.TestCase):
    def test_decodes_full_packet_ltp_best_depth_and_contract_identity(self):
        packet = bytearray(162)
        packet[0] = 8
        struct.pack_into("<H", packet, 1, 162)
        packet[3] = 2
        struct.pack_into("<i", packet, 4, 98765)
        struct.pack_into("<f", packet, 8, 123.45)
        struct.pack_into("<i", packet, 14, 1792670000)
        struct.pack_into("<i", packet, 22, 54321)
        struct.pack_into("<i", packet, 34, 112233)
        struct.pack_into("<iihhff", packet, 62, 250, 300, 4, 5, 123.4, 123.5)

        quote = parse_full_packet(bytes(packet))

        self.assertEqual(quote["segment"], "NSE_FNO")
        self.assertEqual(quote["security_id"], 98765)
        self.assertAlmostEqual(quote["last_price"], 123.45, places=4)
        self.assertAlmostEqual(quote["best_bid"], 123.4, places=4)
        self.assertAlmostEqual(quote["best_ask"], 123.5, places=4)
        self.assertEqual(quote["bid_quantity"], 250)
        self.assertEqual(quote["ask_quantity"], 300)
        self.assertEqual(quote["volume"], 54321)
        self.assertEqual(quote["open_interest"], 112233)
        self.assertIsNotNone(quote["last_trade_at"])

    def test_ignores_unrecognized_or_short_packets(self):
        self.assertIsNone(parse_full_packet(b"\x08"))
        packet = bytearray(162)
        packet[0] = 2
        packet[3] = 2
        struct.pack_into("<i", packet, 4, 98765)
        self.assertIsNone(parse_full_packet(bytes(packet)))

    def test_quote_samples_are_scoped_to_the_recorder_session(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = connect(Path(directory) / "ledger.sqlite3")
            try:
                recorder = DhanQuoteRecorder(
                    Path(directory) / "ledger.sqlite3",
                    "client",
                    "token",
                    dt.date(2026, 10, 22),
                )
                packet = bytearray(162)
                packet[0] = 8
                struct.pack_into("<H", packet, 1, 162)
                packet[3] = 2
                struct.pack_into("<i", packet, 4, 98765)
                recorder._save_packet(connection, bytes(packet))
                row = connection.execute(
                    "SELECT session_date, instrument_segment, security_id FROM paper_market_ticks"
                ).fetchone()
                self.assertEqual(row["session_date"], "2026-10-22")
                self.assertEqual(row["instrument_segment"], "NSE_FNO")
                self.assertEqual(row["security_id"], 98765)
            finally:
                connection.close()

    def test_resolved_trades_remain_subscribed_until_session_end(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = connect(Path(directory) / "ledger.sqlite3")
            try:
                performance_snapshot = {
                    "scan_time": "2026-10-22 09:20:20",
                    "results": [{
                        "verdict": "BUY_CE",
                        "symbol": "AAA",
                        "side": "CE",
                        "ltp": 100,
                        "plan": {"entry": 100, "stop": 95, "t1": 105, "t2": 110},
                        "option": {"security_id": 456},
                    }],
                }
                performance_date = dt.date(2026, 10, 22)
                record_buy_recommendations(
                    connection, "breakout", performance_snapshot,
                    {"AAA": 123}, performance_date,
                )
                connection.execute(
                    "UPDATE paper_trades SET status = 'stop_before_t1'"
                )
                recorder = DhanQuoteRecorder(
                    Path(directory) / "ledger.sqlite3",
                    "client",
                    "token",
                    performance_date,
                )

                self.assertEqual(
                    recorder._instrument_list(connection),
                    {("NSE_EQ", 123), ("NSE_FNO", 456)},
                )
            finally:
                connection.close()

    def test_instrument_list_subscribes_to_open_multiframe_signal_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ledger.sqlite3"
            connection = connect(database)
            try:
                signal_path = Path(directory) / (
                    "daily_hourly_15m_signals_2026-10-22.jsonl"
                )
                signal_path.write_text(
                    json.dumps({
                        "status": "paper_entry_signal_sent",
                        "underlying_security_id": 123,
                        "option": {"security_id": 456},
                    }) + "\n",
                    encoding="utf-8",
                )
                recorder = DhanQuoteRecorder(
                    database, "client", "token", dt.date(2026, 10, 22)
                )

                self.assertEqual(
                    recorder._instrument_list(connection),
                    {("NSE_EQ", 123), ("NSE_FNO", 456)},
                )
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
