import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean

def tracking_error(benchmark: pd.Series, portfolio: pd.Series) -> float:
    """
        Function to compute tracking error
    """
    return (portfolio - benchmark).std()


def annualised_vol(ret):
    """
        Function to convert daily volatility to annualised volatility
    """
    return ret.std() * np.sqrt(252)


def annualised_ret(ret):
    """
        Geometric Average of the returns of each year
    """
    return gmean((1 + ret).resample("1y").prod()) - 1


def sharpe_ratio(ret):
    return np.round(annualised_ret(ret) / annualised_vol(ret), 5)


def drawdown(ret, title):
    wealth_index = (1 + ret).cumprod()
    previous_peak = wealth_index.cummax()
    drawdowns = ((wealth_index - previous_peak) / previous_peak)
    pd.DataFrame({
        "wealth": wealth_index,
        "peak": previous_peak,
        "drawdown": drawdowns
    }).plot(figsize=(15, 10), title="Drawdown of " + title)
    return np.round(abs(drawdowns.min() * 100), 2)
