from trading_algo.base import TradingAlgo
import pandas as pd
import numpy as np
import gurobipy as gb


class CAPM_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("CAPM Weights")
        self.weights = None
        self.expected_returns = None
        self.sigma = None
        self.lookback = None
        self.max_allocation = None
        self.max_leverage = None
        self.stock_data = None
        self.tics = None

    def init_params(self,
                    dataset: pd.DataFrame,
                    lever: float = 0.0,
                    max_cap: float = 0.45,
                    lookback: str = "1 y") -> None:
        self.stock_data = dataset
        self.max_leverage = lever
        self.max_allocation = max_cap
        self.lookback = lookback
        return

    def calculate_inputs(self, date: pd.Timestamp) -> None:
        start_date = date - pd.Timedelta(self.lookback)
        data = self.stock_data[(self.stock_data.index >= start_date) & (self.stock_data.index <= date)]
        data = data.pct_change()[1:]

        self.sigma = np.cov(data.cov())
        self.expected_returns = np.array(data.mean())
        return

    def maximise_sharpe(self) -> None:
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

        # print_equations([],A,sense,b)
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
        if self.tics is None:
            self.tics = price.index.tolist()
        self.calculate_inputs(date)
        self.maximise_sharpe()
        weights = pd.Series(self.weights, index=price.index) * investment  # .apply(math.floor)
        weights = weights.divide(price["close"])
        return weights

    def get_weights(self) -> np.array:
        return self.weights
