from enum import Enum
from typing import Union


class CompositePortfolio:
    def __init__(self):
        self._name: Union[str, None] = None  # Portfolio Identifier
        self._equity: Union[dict, int] = {}  # Equity Instruments:  {'ticker': #shares} or just #shares
        self._debt: Union[dict, int] = {}  # Debt Instruments: {'ticker': #shares} or just #shares
        self._alpha: float = 0  # Portfolio Alpha
        self._beta: float = 1  # Portfolio Beta
        self._sharpe: float = 0  # Portfolio Sharpe
        self._benchmark: str = '^GSPY'  # default index. Use ticker of index, not ETF
        self._valuation: float = 0  # Total Portfolio Valuation
        self._cash_at_hand: float = 0  # Free Cash Reserves
