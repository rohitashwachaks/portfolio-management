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

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        wt = pd.Series(self.weights, index=price.index) * investment  # .apply(math.floor)
        wt = wt.divide(price["close"])
        return wt
