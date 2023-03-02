import pandas as pd
from trading_algo.base import TradingAlgo


class Constant_weight_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("Asset Class Weighted (Constant)")
        pass

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=price.index) * investment  # .apply(math.floor)
        weights = weights.divide(price["close"])
        return weights
