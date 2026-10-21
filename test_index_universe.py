import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import breakout_scanner as scanner


class IndexUniverseTests(unittest.TestCase):
    def test_index_mapping_has_verified_broad_mid_small_and_bank_indices(self):
        self.assertEqual(
            scanner.INDEX_CSV,
            {
                "nifty50": "ind_nifty50list.csv",
                "niftynext50": "ind_niftynext50list.csv",
                "nifty100": "ind_nifty100list.csv",
                "nifty200": "ind_nifty200list.csv",
                "nifty500": "ind_nifty500list.csv",
                "niftymidcap50": "ind_niftymidcap50list.csv",
                "niftymidcap100": "ind_niftymidcap100list.csv",
                "niftymidcap150": "ind_niftymidcap150list.csv",
                "niftysmallcap50": "ind_niftysmallcap50list.csv",
                "niftysmallcap100": "ind_niftysmallcap100list.csv",
                "niftysmallcap250": "ind_niftysmallcap250list.csv",
                "niftybank": "ind_niftybanklist.csv",
            },
        )

    @patch.object(scanner.requests, "get")
    def test_fetch_nifty_bank_accepts_a_valid_constituent_list_under_40(self, get):
        response = Mock()
        response.text = "Company Name,Industry,Symbol,Series,ISIN Code\n" + "".join(
            f"Bank {index},Financial Services,BANK{index},EQ,ISIN{index}\n"
            for index in range(12)
        )
        get.return_value = response

        symbols = scanner.fetch_index_symbols("niftybank")

        self.assertEqual(len(symbols), 12)
        self.assertEqual(symbols[0], "BANK0")
        self.assertEqual(get.call_args.args[0],
                         "https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv")

    @patch.object(scanner, "fetch_index_symbols")
    def test_combined_indices_are_deduplicated_and_filtered_to_stock_options(self, fetch):
        fetch.side_effect = {
            "nifty50": ["AAA", "BBB"],
            "niftysmallcap250": ["BBB", "CCC"],
        }.__getitem__
        eq = pd.DataFrame({"SYMBOL": ["AAA", "BBB", "CCC"], "SID": [101, 102, 103]})
        options = pd.DataFrame({"UNDERLYING": ["AAA", "CCC"]})

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(scanner, "OUT", Path(temp_dir)):
                universe = scanner.build_universe(
                    "nifty50,niftysmallcap250", dt.date(2026, 10, 20), eq, options
                )

        self.assertEqual(universe, {"AAA": 101, "CCC": 103})
        self.assertEqual(fetch.call_args_list[0].args, ("nifty50",))
        self.assertEqual(fetch.call_args_list[1].args, ("niftysmallcap250",))


if __name__ == "__main__":
    unittest.main()
