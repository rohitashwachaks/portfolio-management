import pandas as pd
import yfinance as yf


class Portfolio:
    def __init__(self, target: str,
                 tickerset: set,
                 trading_algo: str,
                 investment: int = 100,
                 rebalance: int = 1,
                 reconstitute: int = 100) -> None:
        """
            Initialise portfolio
        """
        self.IsWeightInitialised = False
        self.target = target
        self.ticker_set = tickerset
        self.tickers = []

        print("Building Portfolios and Initialising Trading Algorithms")
        trade_module = __import__(f"trading_algo.{str.lower(trading_algo)}", fromlist=[trading_algo])
        algo = getattr(trade_module, trading_algo)
        self.trading_algo = algo()

        self.start_nav = investment
        self.prev_day_nav = investment
        self.current_nav = investment
        self.dividends = pd.Series([], dtype=float)

        self.weights = pd.Series([], dtype=float)
        # Equally balanced portfolio. Should Contain number of stocks owned. Ideally, Whole Numbers
        self.last_rebalancing_date = pd.to_datetime("01-01-1990")  # First rebalancing at 01st, Jan
        self.rebalancing_interval = rebalance  # in month
        self.last_reconstitution_date = pd.to_datetime("01-01-1990")  # First reconstitution at 01st, Jan
        self.reconstitution_interval = reconstitute  # in years
        pass

    def update_tickerlist(self, date: pd.Timestamp) -> None:
        self.tickers = self.ticker_set[date.year]
        return

    def isUpdateAllowed(self, date: pd.Timestamp, mode: str = "rebalance") -> bool:
        """
            Calculate Number of months between two dates
        """
        diff = False
        if mode == "rebalance":
            diff = (date.year - self.last_rebalancing_date.year) * 12 + date.month - self.last_rebalancing_date.month
            diff = diff >= self.rebalancing_interval
        elif mode == "reconstitute":
            diff = (
                           date.year - self.last_reconstitution_date.year) * 12 + date.month - self.last_reconstitution_date.month
            diff = diff >= self.reconstitution_interval
        return diff

    def reconstitute(self, price: pd.DataFrame, date: pd.Timestamp, tickerlist: list) -> None:
        """
            Function to Reconstitute our portfolio.
            Checks:
                - weights updated only after "Reconstitution" period has passed
        """
        print("Reconstitution on: " + date.strftime("%d-%b-%Y"))

        if self.isUpdateAllowed(date=date, mode="reconstitute"):
            # Convert weights to number of shares in portfolio
            # Insert reconstitution code here
            if self.last_reconstitution_date != pd.to_datetime("01-01-1990"):
                self.current_nav = self.weights.multiply(price["close"]).sum() + self.dividends.sum()
                self.prev_day_nav = self.current_nav

            self.update_tickerlist(date)
            self.dividends = pd.Series([0] * len(self.tickers), index=self.tickers)
            self.weights = pd.Series([1] * len(self.tickers), index=self.tickers)

            self.IsWeightInitialised = False

            price = price[price.apply(lambda x: (x.name in self.tickers), axis=1)]

            if price.shape[0] == 0:
                return

            self.rebalance(price=price, date=date)
        else:
            raise Exception("Reconstitution Interval not passed.\n\t\
                            Last Reconstituted on: {last_recons}\n\t\
                            Requesting Reconstitution on: {curr_recons}".format(
                last_recons=self.last_reconstitution_date.strftime("%d-%b-%Y"),
                curr_recons=date.strftime("%d-%b-%Y")))
        self.last_reconstitution_date = date
        return

    def rebalance(self, price, date: pd.Timestamp = None) -> None:
        """
            Function to update the Number of shares held by our portfolio.
            Checks:
                - Do ticker lists match?
                - weights updated only after "rebalance" period has passed
        """
        print("\tRebalancing on: " + date.strftime("%d-%b-%Y"))

        if self.isUpdateAllowed(date=date, mode="rebalance"):
            difference_set = {x for x in price.index if x not in set(self.tickers)}
            if len(difference_set) == 0:

                if self.IsWeightInitialised:
                    self.dividends = pd.Series([0.0] * len(self.tickers), index=self.tickers)
                    self.prev_day_nav = self.current_nav
                    self.current_nav = self.weights.multiply(price["close"]).sum() + self.dividends.sum()

                self.weights = self.trading_algo.run(price, self.current_nav, date)
                self.last_rebalancing_date = date
                self.IsWeightInitialised = True
            else:
                raise Exception(" Input and Portfolio Tickers Not Matching. Net mismatch: {mismatch}\n".format(
                    mismatch=len(difference_set)) + ", ".join(list(difference_set)));
        else:
            raise Exception("rebalance Interval not passed.\n\t\
                            Last rebalanced on: {last_rebal}\n\t\
                            Requesting rebalance on: {curr_rebal}".format(
                last_rebal=self.last_rebalancing_date.strftime("%d-%b-%Y"),
                curr_rebal=date.strftime("%d-%b-%Y")))
        return

    def run(self, price: pd.DataFrame, date: pd.Timestamp, tickerlist: list = None) -> tuple:
        """
            This function will be run on every date
        """
        price.fillna(0, inplace=True)
        self.dividends += price["div"]

        if sum(price.shares) != 0:
            if self.isUpdateAllowed(date, mode="reconstitute"):
                self.reconstitute(date=date, price=price, tickerlist=tickerlist)
            else:
                price = price[price.apply(lambda x: (x.name in self.tickers), axis=1)]

                if price.shape[0] == 0:
                    return -1, -1
                if self.isUpdateAllowed(date):
                    self.rebalance(date=date, price=price)
                else:
                    self.prev_day_nav = self.current_nav
                    self.current_nav = self.weights.multiply(price["close"]).sum() + self.dividends.sum()

        return self.current_nav, self.get_returns()

    def get_nav(self, price: pd.DataFrame) -> float:
        nav = self.weights.multiply(price["close"]).sum() + self.dividends.sum()
        return nav

    def get_returns(self) -> float:
        """
            Function to calculate returns of our portfolio
        """
        return (self.current_nav - self.prev_day_nav + self.dividends.sum()) / self.prev_day_nav

    def echo(self) -> None:
        print("Target Index:", self.target)
        print("Trading Algorithm:", self.trading_algo.identifier)
        print("Starting Investment:", self.start_nav)
        print("Current Investment:", self.current_nav)
        if len(self.tickers) > 0:
            print("Constituent Indices:", ", ".join(self.tickers[0:5] + ["\b\b ..."]))
        if len(self.weights) > 0:
            print("Constituent Weights:", self.weights[0:5])
        pass


if __name__ == "__main__":
    start_date = pd.to_datetime("2010-01-01")
    end_date = pd.to_datetime("2020-12-31")
    portfolio_tickers = ["SPY", "ACWX", "VCIT", "IGOV", "USO"]

    benchmark_weights = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10], index=portfolio_tickers)

    stocks = yf.Tickers(" ".join(portfolio_tickers))

    stock_data = stocks.download(" ".join(portfolio_tickers), start=start_date, end=end_date)[
        ["Close", "Dividends"]]  # .dropna()
    stock_data.rename({"Close": "close", "Dividends": "div"}, axis=1, inplace=True)
    stock_data["close"].plot(figsize=(30, 10))

    for tic in portfolio_tickers:
        stock_data[("shares", tic)] = pd.Series([1] * len(stock_data.index), index=stock_data.index)

    ticker_set = dict()
    for year in range(start_date.year, end_date.year + 1):
        ticker_set[year] = portfolio_tickers

    Portfolio(target="Baseline",
              tickerset=ticker_set, investment=100,
              trading_algo="Constant_weight_Algo",
              rebalance=1, reconstitute=100)
#-------------------------------------------------------

import pandas as pd
from trading_algo.base import TradingAlgo
import numpy as np
import gurobipy as gb
import cplex


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

    def maximise_sharpe_cplex(self):
        variables = len(self.tics)
        constraints = 1 + (2 * variables)

        # Defining Model's Decision Variables
        maxSharpe_model: cplex.Cplex = cplex.Cplex()
        y = [maxSharpe_model.variables.add(obj=0.0, lb=0.0, ub=1.0, types=maxSharpe_model.variables.type.continuous) for i in range(variables)]


        # Defining Model's Objective Function (Minimize Risk)
        risk = y @ self.sigma @ y
        maxSharpe_model.setObjective(risk, sense=gb.GRB.MINIMIZE)

        # Define the objective function
        maxSharpe_model.objective.set_sense(maxSharpe_model.objective.sense.maximize)
        maxSharpe_model.variables.set_lower_bounds(0, 0.0)
        maxSharpe_model.objective.set_linear([(i, returns[i]) for i in range(num_assets)])
        maxSharpe_model.objective.set_quadratic([(i, j, covariance[i][j]) for i in range(num_assets) for j in range(num_assets)])


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
