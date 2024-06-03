###################
# tests for Market #
###################
import unittest

import pandas as pd

from finquant.market import Market
from finquant.portfolio import build_portfolio


class TestMarket(unittest.TestCase):
    """
    TODO: ...
    """

    def setUp(self):
        d = {
            0: {"Name": "GOOG", "Allocation": 20},
            1: {"Name": "AMZN", "Allocation": 10},
            2: {"Name": "MCD", "Allocation": 15},
            3: {"Name": "DIS", "Allocation": 18},
            4: {"Name": "TSLA", "Allocation": 48},
        }

        self.pf_allocation = pd.DataFrame.from_dict(d, orient="index")
        self.names_yf = self.pf_allocation["Name"].values.tolist()

        # dates can be set as datetime or string, as shown below:
        self.start_date = "2018-01-01"
        self.end_date = "2023-01-01"

    def test_market(self):
        """
        TODO: ...
        """

        pf = build_portfolio(
            names=self.names_yf,
            pf_allocation=self.pf_allocation,
            start_date=self.start_date,
            end_date=self.end_date,
            data_api="yfinance",
            market_index="^GSPC",
        )
        assert isinstance(pf.market_index, Market)

        assert pf.market_index.name == "^GSPC"

        assert pf.beta is not None

        assert pf.rsquared is not None

        assert pf.treynor is not None
