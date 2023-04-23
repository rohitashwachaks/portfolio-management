import pandas as pd

from trading_algo.base import TradingAlgo


class SNP500_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("MarketCap Weighted Portfolio")
        pass

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        price["cap"] = price["shares"] * price["close"]
        nav = sum(price["cap"])

        weights = (((price["cap"] / nav) * investment) / price["close"])  # .apply(math.floor)
        return weights
