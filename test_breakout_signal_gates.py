import unittest

import breakout_scanner


class BreakoutSignalGateTests(unittest.TestCase):
    def test_live_price_must_remain_beyond_call_trigger(self):
        self.assertTrue(breakout_scanner._live_price_beyond_trigger("CE", 101, 100))
        self.assertFalse(breakout_scanner._live_price_beyond_trigger("CE", 100, 100))
        self.assertFalse(breakout_scanner._live_price_beyond_trigger("CE", 99, 100))

    def test_live_price_must_remain_beyond_put_trigger(self):
        self.assertTrue(breakout_scanner._live_price_beyond_trigger("PE", 99, 100))
        self.assertFalse(breakout_scanner._live_price_beyond_trigger("PE", 100, 100))
        self.assertFalse(breakout_scanner._live_price_beyond_trigger("PE", 101, 100))


if __name__ == "__main__":
    unittest.main()
