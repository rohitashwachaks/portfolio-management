import json

import pandas as pd
from trading_algo.base import TradingAlgo
import numpy as np
import gurobipy as gb
import cplex
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from scipy.stats.mstats import gmean


class retrospective_sharpe_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("Retrospective Max Weights")
        self.expected_returns = None
        self.sigma = None
        self.max_allocation = None
        self.max_leverage = None
        self.tics = None
        self.stock_data = None
        self.weights = None
        return

    def init_params(self, ticker_list: list,
                    dataset=pd.DataFrame,
                    lever: float = 0.0,
                    max_cap: float = 0.45) -> None:
        self.stock_data = dataset.pct_change()[1:]
        self.tics = ticker_list
        self.max_leverage = lever
        self.max_allocation = max_cap

        self.sigma = np.cov(self.stock_data.cov())
        self.expected_returns = np.array(self.stock_data.mean())

        self.maximise_sharpe()
        return

    def maximise_sharpe(self):
        variables = len(self.tics)
        constraints = 1 + (2 * variables)

        # Defining Model's Decision Variables
        maxSharpe_model = cplex.Cplex()
        y = maxSharpe_model.variables.add(obj=[0.0] * variables,
                                          lb=[0.0] * variables,
                                          ub=[1.0] * variables,
                                          types=['C'] * variables,
                                          names=[f'stock{_}' for _ in range(variables)])

        # y = [maxSharpe_model.variables.add(obj=0.0, lb=0.0, ub=1.0, types=maxSharpe_model.variables.type.continuous) for i in range(variables)]

        # Defining Model's Objective Function (Minimize Risk)
        risk = np.tril((self.sigma + self.sigma.T)-(np.diag(self.sigma)*np.eye(self.sigma.shape[0])))
        quadratic_objective = []
        for row_index, row in enumerate(risk):
            for col_index, loading in enumerate(row):
                if loading != 0:
                    quadratic_objective.append(f'{loading}*stock_{row_index}*stock_{col_index}')
        quadratic_objective = ' + '.join(quadratic_objective)
        maxSharpe_model.objective.set_quadratic(quadratic_objective)
        maxSharpe_model.objective.set_sense(maxSharpe_model.objective.sense.minimize)

        # Defining Constraints
        A = np.zeros((constraints, variables))
        A[0] = self.expected_returns  # Sum of weights = 1
        A[1:variables + 1] = np.eye(variables) - np.ones((variables, variables)) * (-self.max_leverage)  # investment>0
        A[-variables:] = np.eye(variables) - np.ones((variables, variables)) * self.max_allocation  # Sum

        b = np.array([1] + [0] * (constraints - 1))
        sense = np.array(["="] + [">"] * variables + ["<"] * variables)

        maxSharpe_model.linear_constraints.add(lin_expr=A,
                                               senses=sense,
                                               rhs=b)
        maxSharpe_model.write('maxSharpe_model.lp')
        maxSharpe_model.solve()

        print("Obj Value:", maxSharpe_model.solution.get_objective_value())
        print("Values of Decision Variables:", maxSharpe_model.solution.get_values())
        maxSharpe_model.solution.write('maxSharpe_model.txt')

        # Optimize Model
        maxSharpe_model.Params.OutputFlag = 0
        maxSharpe_model.optimize()

        try:
            optimal_values = y.x
            self.weights = np.round(optimal_values / optimal_values.sum(), 2)
        except Exception as e:
            print("Failed to find Optimal Weights")
        return

    def maximise_sharpe_gb(self):
        variables = len(self.tics)
        constraints = 1 + (2 * variables)

        # Defining Model's Decision Variables
        maxSharpe_model = gb.Model()
        y = maxSharpe_model.addMVar(variables)

        # Defining Model's Objective Function (Minimize Risk)
        risk = y @ self.sigma @ y
        maxSharpe_model.setObjective(risk, sense=gb.GRB.MINIMIZE)

        # Defining Constraints
        A = np.zeros((constraints, variables))
        A[0] = self.expected_returns  # Sum of weights = 1
        A[1:variables + 1] = np.eye(variables) - np.ones((variables, variables)) * (-self.max_leverage)
        A[-variables:] = np.eye(variables) - np.ones((variables, variables)) * self.max_allocation

        b = np.array([1] + [0] * (constraints - 1))
        sense = np.array(["="] + [">"] * variables + ["<"] * variables)

        maxSharpe_model.addMConstrs(A, y, sense, b)

        # Optimize Model
        maxSharpe_model.Params.OutputFlag = 0
        maxSharpe_model.optimize()

        try:
            optimal_values = y.x
            self.weights = np.round(optimal_values / optimal_values.sum(), 2)
        except Exception as e:
            print("Failed to find Optimal Weights")
        return

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        wt = pd.Series(self.weights, index=price.index) * investment  # .apply(math.floor)
        wt = wt.divide(price["close"])
        return wt


# --------------------------------------------------------
warnings.filterwarnings("ignore")


# Helper Functions & Initialising Scripts

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
    df = pd.DataFrame({
        "wealth": wealth_index,
        "peak": previous_peak,
        "drawdown": drawdowns
    }).plot(figsize=(15, 10), title="Drawdown of " + title)
    return np.round(abs(drawdowns.min() * 100), 2)


if __name__ == "__main__":
    prt = __import__("portfolio")
    Portfolio = getattr(prt, "Portfolio")

    # region Online
    # # Read Portfolios and trading Strategies
    # ## Initialising
    # start_date = pd.to_datetime("2010-01-01")
    # end_date = pd.to_datetime("2020-12-31")
    # portfolio_tickers = ["SPY", "ACWX", "VCIT", "IGOV", "USO"]
    #
    # ## Yahoo API
    # stocks = yf.Tickers(" ".join(portfolio_tickers))
    #
    # stock_data = stocks.download(" ".join(portfolio_tickers), start=start_date, end=end_date)[
    #     ["Close", "Dividends"]]  # .dropna()
    # stock_data.rename({"Close": "close", "Dividends": "div"}, axis=1, inplace=True)
    #
    # for tic in portfolio_tickers:
    #     stock_data[("shares", tic)] = pd.Series([1] * len(stock_data.index), index=stock_data.index)
    #
    # # portfolio_tickers = stock_data["close"].columns
    #
    # stock_data.to_csv('../data/stock_data_close.csv')
    # endregion Online

    stock_data = pd.read_csv('../data/stock_data_close.csv', index_col=0, header=[0, 1], parse_dates=True)
    portfolio_tickers = stock_data['close'].columns
    start_date, end_date = stock_data.index.sort_values()[[0, -1]].tolist()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    ticker_dict = dict()
    for year in range(start_date.year, end_date.year + 1):
        ticker_dict[year] = portfolio_tickers

    # BenchMark Portfolio
    ## Simulation
    from trading_algo import *

    benchmark_weights = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10], index=portfolio_tickers)

    benchmark_portfolio = Portfolio(target="Baseline",
                                    tickerset=ticker_dict, investment=100,
                                    trading_algo="Constant_weight_Algo",
                                    rebalance=3, reconstitute=99)

    benchmark_portfolio.echo()
    benchmark_val = pd.Series([], dtype=float)
    benchmark_returns = pd.Series([], dtype=float)

    for date in stock_data.index:
        tmpdf = stock_data.loc[date]
        tmpdf = tmpdf.unstack().T
        valuation, returns = benchmark_portfolio.run(date=date, price=tmpdf, tickerlist=portfolio_tickers)
        if (valuation, returns) == (-1, -1):
            print(benchmark_val)
            print(benchmark_val.index[-1], date.date(), valuation, returns)
            continue
        benchmark_val[date] = valuation
        benchmark_returns[date] = returns

    # Retrospective Portifolio
    ## Simulation
    retrospective_portfolio = Portfolio(target="Retrospective Portfolio",
                                        tickerset=ticker_dict, investment=100,
                                        trading_algo="retrospective_sharpe_Algo", rebalance=1)

    retrospective_portfolio.trading_algo.init_params(dataset=stock_data["close"], ticker_list=portfolio_tickers,
                                                     lever=0.3)
    retrospective_portfolio.echo()
