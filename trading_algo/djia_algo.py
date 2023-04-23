import pandas as pd
from trading_algo.base import TradingAlgo


class DJIA_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("Price Cap Weighted")
        pass

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        total_price = sum(price["close"])

        weights = [investment / total_price] * len(price.index)
        weights = pd.Series(weights, index=price.index)  # .apply(math.floor)
        return weights
