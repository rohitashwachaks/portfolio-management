import cplex
from trading_algo.base import TradingAlgo
import pandas as pd
import numpy as np


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
        self.rf = 0.0
        return

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

    def maximise_sharpe(self):
        try:
            variables = len(self.tics)
            constraints = 1 + (2 * variables)

            # Defining Model's Decision Variables
            maxSharpe_model = cplex.Cplex()
            maxSharpe_model.set_log_stream(None)
            maxSharpe_model.set_error_stream(None)
            maxSharpe_model.set_warning_stream(None)
            maxSharpe_model.set_results_stream(None)
            maxSharpe_model.parameters.optimalitytarget.set(3)

            maxSharpe_model.variables.add(obj=[0.0] * variables,
                                          lb=[0.0] * variables,
                                          ub=[1.0] * variables,
                                          types=[maxSharpe_model.variables.type.continuous] * variables,
                                          names=[f'stock{_}' for _ in range(variables)])

            # Defining Model's Objective Function (Minimize Risk)
            quadratic_objective = []
            for row_index, row in enumerate(self.sigma):
                for col_index, loading in enumerate(row):
                    if loading != 0:
                        quadratic_objective.append((row_index, col_index, loading))
            maxSharpe_model.objective.set_quadratic_coefficients(quadratic_objective)
            maxSharpe_model.objective.set_sense(maxSharpe_model.objective.sense.minimize)

            # Defining Constraints
            A = np.zeros((constraints, variables))
            A[0] = (self.expected_returns - self.rf)  # Sum of weights = 1
            A[1:variables + 1] = np.eye(variables) - (np.ones((variables, variables)) * (
                -self.max_leverage))  # investment>0
            A[-variables:] = np.eye(variables) - np.ones((variables, variables)) * self.max_allocation  # Sum

            b = np.array([1.0] + [0.0] * (constraints - 1))
            sense = np.array(["E"] + ["G"] * variables + ["L"] * variables)

            maxSharpe_model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[f'stock{_}' for _ in range(variables)], val=t) for t in A],
                senses=sense, rhs=b)
            maxSharpe_model.solve()

            # maxSharpe_model.write('maxSharpe_model.lp')
            # print("Obj Value:", maxSharpe_model.solution.get_objective_value())
            # print("Values of Decision Variables:", maxSharpe_model.solution.get_values())
        except Exception as e:
            print(e)

        try:
            optimal_values = np.array(maxSharpe_model.solution.get_values())
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
