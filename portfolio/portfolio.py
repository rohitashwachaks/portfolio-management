import json
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Union, List, Dict
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from trading_algo.base import TradingAlgo

GSPY_ticker = '^GSPY'


@dataclass
class AssetBaseClass:
    ticker: str
    shares: int = field(default=0)
    benchmark: str = field(default=GSPY_ticker)

    valuation: float = field(init=False)
    alpha: float = field(init=False)
    beta: float = field(init=False)
    sharpe: float = field(init=False)


@dataclass
@dataclass_json
class Asset(AssetBaseClass):
    # region Internal Functions
    def __init__(self, ticker: str,
                 shares: int = 0,
                 benchmark: str = GSPY_ticker,
                 window: Union[str, int, None] = None,
                 end_date: date = datetime.today().date()):
        super().__init__(ticker=ticker,
                         shares=shares,
                         benchmark=benchmark)

        self.window = window
        self._end_date = end_date

        if self.window == 'YTD':
            self._start_date = datetime(year=self._end_date.year, month=1, day=1).date()
        elif isinstance(self.window, int):
            self._start_date = self._end_date - timedelta(days=self.window)
        else:
            self._start_date = datetime(year=1970, month=1, day=1).date()

        self._stock_data = self._fetch_price_()

    def _fetch_price_(self, visualise: bool = False) -> pd.DataFrame:
        stocks = yf.Ticker(self.ticker)

        stock_data = stocks.history(start=self._start_date,
                                    end=self._end_date)  # [["Close", "Dividends"]]  # .dropna()
        stock_data.rename({"Date": "Date",
                           "Open": "open",
                           "Close": "close",
                           "Volume": "volume",
                           "Dividends": "dividends",
                           "Stock Splits": "splits"}, axis=1, inplace=True)

        stock_data = self._fill_missing_dates_(stock_data)
        if visualise:
            stock_data["close"].plot(figsize=(30, 10))

        return stock_data

    def _fill_missing_dates_(self, fin_data: pd.DataFrame) -> pd.DataFrame:
        # Reindex the DataFrame with a complete date range
        date_range = pd.date_range(start=self._start_date,
                                   end=self._end_date, freq='D')
        date_range = [dt.strftime('%Y-%m-%d') for dt in date_range]
        fin_data = fin_data.reindex(date_range)
        fin_data.ffill(inplace=True)
        fin_data.bfill(inplace=True)  # fill in for dates Jan 1st and 2nd (because market closed due to new years

        # Forward fill missing values
        return fin_data.ffill()

    # endregion Internal Functions

    # region Properties

    @property
    def end_date(self) -> str:
        # Assuming 'end_date' is in the format 'YYYY-MM-DD'
        return self._end_date.strftime('%Y-%m-%d')

    @property
    def start_date(self) -> str:
        return self._start_date.strftime('%Y-%m-%d')

    @property
    def valuation(self) -> float:
        """
        Multiply number of shares/holding
        :return:
        """
        stock_price = self.price()
        return self.shares * stock_price

    @property
    def alpha(self) -> float:
        """
        Asset Alpha
        :return:
        """
        return 0

    @property
    def beta(self) -> float:
        """
        Asset Beta
        :return:
        """
        return 1

    @property
    def sharpe(self) -> float:
        """
        Asset Sharpe
        :return:
        """
        return 0

    # endregion Properties

    # region Function
    def price(self, price_date: str = None) -> float:
        if price_date is None:
            price_date = self.end_date

        return self._stock_data['close'][price_date]

    # endregion Function


@dataclass
class PortfolioBaseClass:
    name: str
    assets: Dict[str, Asset] = field(default_factory=defaultdict(Asset))
    window: Union[None, int] = field(default=None)
    weight: float = field(init=False)

@dataclass
class BasePortfolioClass:
    def __init__(self, name: str,
                 ) -> None:
        self.name: str = name
        self.asset: Asset = Asset()
    name: str
    assets: Asset = field(default_factory=Asset)
    window: Union[None, int] = field(default=None)
    weights: np.array = field(init=False)


@dataclass
@dataclass_json
class Portfolio(PortfolioBaseClass):
    def __init__(self, name: str,
                 target: str,
                 nav: int = 100,
                 rebalance: int = 1,
                 reconstitute: int = 100,
                 assets: Union[List[str], Dict[str, Asset], None] = None,
                 window: Union[None, int] = None,
                 trading_algo: Union[None, TradingAlgo] = None):
        """
        Portfolio class
        :param name: Name of the portfolio
        :param target: Benchmark target
        :param trading_algo: Trading Algo
        :param nav: Net Asset Value
        :param assets: Dictionary of Assets
        :param rebalance: Number of months to rebalance the portfolio
        :param reconstitute: Number of months to reconstitute the portfolio
        """
        self.IsWeightInitialised: bool = False

        if isinstance(assets, List):
            assets = {ticker: Asset(ticker) for ticker in assets}

        super().__init__(name=name,
                         assets=assets,
                         window=window)

        self.weights: Dict[str, float] = {x: 0 for x in self.assets.keys()}

        self.start_nav = nav
        self.prev_day_nav = nav
        self.current_nav = nav

        self.rebalancing_interval: int = rebalance
        self.reconstitution_interval: int = reconstitute
        self.last_rebalancing_date = datetime('1970-01-01').date()
        self.last_reconstitution_date = datetime('1970-01-01').date()

        # self.trading_algo: TradingAlgo = trading_algo
        print("Building Portfolios and Initialising Trading Algorithms")
        trade_module = __import__(f"trading_algo.{str.lower(trading_algo)}", fromlist=[trading_algo])
        algo = getattr(trade_module, trading_algo)
        self.trading_algo = algo()

        self.target: Asset = Asset(ticker=target,
                                   window=window,
                                   shares=1)

    # region Properties
    @property
    def tickerset(self):
        return self.assets.keys()

    @property
    def weights(self):


    # endregion Properties

    # region Functions

    # def update_tickerlist(self, date: pd.Timestamp) -> None:
    #     self.tickers = self.ticker_set[date.year]
    #     return

    def isUpdateAllowed(self, date: pd.Timestamp, mode: str = "rebalance") -> bool:
        """
        Calculate Number of months between two dates and determine if it's time to update the weights
        :param date:
        :param mode:
        :return:
        """
        diff = False
        if mode == "rebalance":
            diff = (date.year - self.last_rebalancing_date.year) * 12 + date.month - self.last_rebalancing_date.month
            diff = diff >= self.rebalancing_interval
        elif mode == "reconstitute":
            diff = (
                           date.year - self.last_reconstitution_date.year) * 12 + date.month - self.last_reconstitution_date.month
            diff = diff >= self.reconstitution_interval
        return diff

    # TODO
    def reconstitute(self, price: pd.DataFrame, date: pd.Timestamp, tickerlist: list) -> None:
        """
        Function to Reconstitute our portfolio.
            Checks:
                - weights updated only after "Reconstitution" period has passed
        :param price:
        :param date:
        :param tickerlist:
        :return:
        """
        print("Reconstitution on: " + date.strftime("%d-%b-%Y"))

        if self.isUpdateAllowed(date=date, mode="reconstitute"):
            # Convert weights to number of shares in portfolio
            # Insert reconstitution code here
            if self.last_reconstitution_date != pd.to_datetime("01-01-1990"):
                self.current_nav = self.weights.multiply(price["close"]).sum() + self.dividends.sum()
                self.prev_day_nav = self.current_nav

            self.update_tickerlist(date)
            self.dividends = pd.Series([0] * len(self.tickers), index=self.tickers)
            self.weights = pd.Series([1] * len(self.tickers), index=self.tickers)

            self.IsWeightInitialised = False

            price = price[price.apply(lambda x: (x.name in self.tickers), axis=1)]

            if price.shape[0] == 0:
                return

            self.rebalance(price=price, date=date)
        else:
            raise RuntimeError("Reconstitution Interval not passed.\n\t" \
                               "Last Reconstituted on: {last_recons}\n\t" \
                               "Requesting Reconstitution on: {curr_recons}".format(
                last_recons=self.last_reconstitution_date.strftime("%d-%b-%Y"),
                curr_recons=date.strftime("%d-%b-%Y")))
        self.last_reconstitution_date = date

    # def rebalance(self, price, date: pd.Timestamp = None) -> None:
    #     """
    #         Function to update the Number of shares held by our portfolio.
    #         Checks:
    #             - Do ticker lists match?
    #             - weights updated only after "rebalance" period has passed
    #     """
    #     print("\tRebalancing on: " + date.strftime("%d-%b-%Y"))
    #
    #     if self.isUpdateAllowed(date=date, mode="rebalance"):
    #         difference_set = {x for x in price.index if x not in set(self.tickers)}
    #         if len(difference_set) == 0:
    #
    #             if self.IsWeightInitialised:
    #                 self.dividends = pd.Series([0.0] * len(self.tickers), index=self.tickers)
    #                 self.prev_day_nav = self.current_nav
    #                 self.current_nav = self.weights.multiply(price["close"]).sum() + self.dividends.sum()
    #
    #             self.weights = self.trading_algo.run(price, self.current_nav, date)
    #             self.last_rebalancing_date = date
    #             self.IsWeightInitialised = True
    #         else:
    #             raise Exception(" Input and Portfolio Tickers Not Matching. Net mismatch: {mismatch}\n".format(
    #                 mismatch=len(difference_set)) + ", ".join(list(difference_set)));
    #     else:
    #         raise Exception("rebalance Interval not passed.\n\t\
    #                         Last rebalanced on: {last_rebal}\n\t\
    #                         Requesting rebalance on: {curr_rebal}".format(
    #             last_rebal=self.last_rebalancing_date.strftime("%d-%b-%Y"),
    #             curr_rebal=date.strftime("%d-%b-%Y")))
    #     return
    #
    # def run(self, price: pd.DataFrame, date: pd.Timestamp, tickerlist: list = None) -> tuple:
    #     """
    #         This function will be run on every date
    #     """
    #     price.fillna(0, inplace=True)
    #     self.dividends += price["div"]
    #
    #     if sum(price.shares) != 0:
    #         if self.isUpdateAllowed(date, mode="reconstitute"):
    #             self.reconstitute(date=date, price=price, tickerlist=tickerlist)
    #         else:
    #             price = price[price.apply(lambda x: (x.name in self.tickers), axis=1)]
    #
    #             if price.shape[0] == 0:
    #                 return -1, -1
    #             if self.isUpdateAllowed(date):
    #                 self.rebalance(date=date, price=price)
    #             else:
    #                 self.prev_day_nav = self.current_nav
    #                 self.current_nav = self.weights.multiply(price["close"]).sum() + self.dividends.sum()
    #
    #     return self.current_nav, self.get_returns()
    #
    # def get_nav(self, price: pd.DataFrame) -> float:
    #     nav = self.weights.multiply(price["close"]).sum() + self.dividends.sum()
    #     return nav
    #
    # def get_returns(self) -> float:
    #     """
    #         Function to calculate returns of our portfolio
    #     """
    #     return (self.current_nav - self.prev_day_nav + self.dividends.sum()) / self.prev_day_nav
    #
    # def echo(self) -> None:
    #     print("Target Index:", self.target)
    #     print("Trading Algorithm:", self.trading_algo.identifier)
    #     print("Starting Investment:", self.start_nav)
    #     print("Current Investment:", self.current_nav)
    #     if len(self.tickers) > 0:
    #         print("Constituent Indices:", ", ".join(self.tickers[0:5] + ["\b\b "]))
    #     if len(self.weights) > 0:
    #         print("Constituent Weights:", self.weights[0:5])
    #     pass
    #
    # # endregion Functions


if __name__ == '__main__':
    window = 'YTD'
    # region Assets
    aapl = Asset(ticker='AAPL',
                 window=window)

    msft = Asset(ticker='MSFT',
                 window=window)

    tsla = Asset(ticker='TSLA',
                 window=window)

    # endregion Assets

    portfolio = Portfolio(name='Stonks',
                          target='^GSPY',
                          assets={
                              'AAPL': aapl,
                              'MSFT': msft,
                              'TSLA': tsla
                          },
                          nav=1000)
    p
