from enum import Enum
from typing import Union
from dataclasses import dataclass
from portfolio.asset import Asset


@dataclass
class Portfolio(Asset):
    def __init__(self, name: Union[str, None] = None,
                 ticker: Union[str, None] = None,
                 shares: int = 0,
                 benchmark: str = '^GSPY'):
        super().__init__(name=name, ticker=ticker,
                         shares=0, )
        self._name: Union[str, None] = None  # Portfolio Identifier
        self._equity: Union[dict, int] = {}  # Equity Instruments:  {'ticker': #shares} or just #shares
        self._debt: Union[dict, int] = {}  # Debt Instruments: {'ticker': #shares} or just #shares
        # self._alpha: float = 0  # Portfolio Alpha
        # self._beta: float = 1  # Portfolio Beta
        # self._sharpe: float = 0  # Portfolio Sharpe
        self._benchmark: str = benchmark  # default index. Use ticker of index, not ETF
        self._valuation: float = 0  # Total Portfolio Valuation
        self._cash_at_hand: float = 0  # Free Cash Reserves
