from typing import Union
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Asset:
    def __init__(self, name: Union[str, None] = None,
                 ticker: Union[str, None] = None,
                 shares: int = 0,
                 benchmark: str = '^GSPY'):
        """
        Initialise Asset. Base class extended by equity, debt etc instruments. Fundamental building block of every Portfolio
        :param name: Asset Identifier
        :param ticker: Asset Ticker
        :param shares: Number of shares held. >0 for long. <0 for short. Whole shares only (for now)
        :param benchmark: Ticker of index, not ETF. Default: ^GSPY
        """
        self._name: Union[str, None] = name
        self._ticker: Union[str, None] = ticker
        self._shares: Union[float, None] = shares
        self._benchmark: str = benchmark

        assert self._name is not None
        assert self._ticker is not None
        assert self._shares is not None and isinstance(self._shares, int)
        assert self._benchmark is not None

    # region Properties
    @property
    def name(self):
        return self._name

    @property
    def ticker(self):
        return self._ticker

    @property
    def shares(self):
        return self._shares

    @property
    def benchmark(self):
        return self._benchmark

    # endregion Properties

    # region Functions
    def valuation(self, date: Union[datetime, None] = None,
                  price: Union[float, None] = None) -> None:
        """
        Multiply number of shares/holding
        :param date:
        :param price:
        :return:
        """
        _valuation = None
        if date is not None:
            _valuation = None
        if price is not None:
            _valuation = self._shares * price
        return _valuation

    def alpha(self, start_date: datetime, end_date: datetime) -> float:
        """
        Asset Alpha
        :return:
        """
        return 0

    def beta(self, start_date: datetime, end_date: datetime) -> float:
        """
        Asset Beta
        :return:
        """
        return 1

    def sharpe(self, start_date: datetime, end_date: datetime) -> float:
        """
        Asset Sharpe
        :return:
        """
        return 0

    # endregion Functions
