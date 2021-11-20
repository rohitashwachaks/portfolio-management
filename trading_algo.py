import pandas as pd
import numpy as np
import math

class TradingAlgo:
    def __init__(self, name: str) -> None:
        self.identifier = name
        pass

class SNP500_Algo(TradingAlgo):
    def run(self, price: pd.DataFrame,
                investment: float) -> pd.Series(dtype=float):
        '''
            .
        '''
        price["cap"] = price["shares"]*price["close"]
        nav = sum(price["cap"])

        weights = (((price["cap"]/nav)*investment)/price["close"])
        return weights

class DJIA_Algo(TradingAlgo):
    def run(self, price: pd.DataFrame,
                investment: float) -> pd.Series(dtype=float):
        '''
            .
        '''
        total_price = sum(price["close"])

        weights = [investment/total_price]*len(price.index)                             # Equivalent to the above Equation
        weights = pd.Series(weights, index= price.index)
        return weights