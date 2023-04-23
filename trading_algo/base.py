from abc import abstractmethod

import pandas as pd


class TradingAlgo:
    def __init__(self, name: str) -> None:
        self.identifier = name
        pass

    @abstractmethod
    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        pass
