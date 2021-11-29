import pandas as pd
import numpy as np
import gurobipy as gb
import yfinance as yf

class TradingAlgo():  
    def __init__(self, name: str) -> None:
        self.identifier = name
        pass

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        pass


class SNP500_Algo(TradingAlgo):    
    def __init__(self) -> None:
        super().__init__("MarketCap Weighted Portfolio")
        pass

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        price["cap"] = price["shares"]*price["close"]
        nav = sum(price["cap"])

        weights = (((price["cap"]/nav)*investment)/price["close"])#.apply(math.floor)
        return weights


class DJIA_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("Price Cap Weighted")
        pass

    def run(self, price: pd.DataFrame,
                investment: float) -> pd.Series(dtype=float):
        total_price = sum(price["close"])

        weights = [investment/total_price]*len(price.index)
        weights = pd.Series(weights, index= price.index)#.apply(math.floor)
        return weights


class constant_weight_Algo(TradingAlgo):    
    def __init__(self) -> None:
        super().__init__("Asset Class Weighted (Constant)")
        pass
    
    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        weights = pd.Series([0.3,0.25,0.2,0.15, 0.1], index = price.index)*investment#.apply(math.floor)
        weights = weights.divide(price["close"])
        return weights

class retrospective_sharpe_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("Retrospective Max Weights")
        return

    def init_params(self, ticker_list: list,
                    dataset = pd.DataFrame,
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
        constraints = 1+(2*variables)

        # Defining Model's Decision Variables
        maxSharpe_model = gb.Model()
        y = maxSharpe_model.addMVar(variables)
        
        # Defining Model's Objective Function (Minimize Risk)
        risk = y @ self.sigma @ y
        maxSharpe_model.setObjective(risk, sense=gb.GRB.MINIMIZE)

        # Defining Constraints
        A = np.zeros((constraints,variables))
        A[0] = self.expected_returns # Sum of weights = 1
        A[1:variables+1] = np.eye(variables) - np.ones((variables,variables))*(-self.max_leverage)
        A[-variables:] = np.eye(variables) - np.ones((variables,variables))*self.max_allocation
        
        b = np.array([1]+[0]*(constraints-1))
        sense = np.array(["="]+[">"]*variables+["<"]*variables)

        maxSharpe_model.addMConstrs(A, y, sense, b)

        # Optimize Model
        maxSharpe_model.Params.OutputFlag = 0
        maxSharpe_model.optimize()

        try:
            optimal_values = y.x
            self.weights = np.round(optimal_values/optimal_values.sum(), 2)
        except:
            print("Failed to find Optimal Weights")
        return

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        wt = pd.Series(self.weights, index = price.index)*investment#.apply(math.floor)
        wt = wt.divide(price["close"])
        return wt

class capm_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("CAPM Weights")
        self.tics = None
        pass

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

    def calculate_inputs(self, date: pd.Timestamp)->None:
        start_date = date -  pd.Timedelta(self.lookback)
        data = self.stock_data[(self.stock_data.index >= start_date) & (self.stock_data.index <=date)]
        data = data.pct_change()[1:]
        
        self.sigma = np.cov(data.cov())
        self.expected_returns = np.array(data.mean())
        return

    def maximise_sharpe(self)-> None:
        variables = len(self.tics)
        constraints = 1+(2*variables)

        # Defining Model's Decision Variables
        maxSharpe_model = gb.Model()
        y = maxSharpe_model.addMVar(variables)
        
        # Defining Model's Objective Function (Minimize Risk)
        risk = y @ self.sigma @ y
        maxSharpe_model.setObjective(risk, sense=gb.GRB.MINIMIZE)

        # Defining Constraints
        A = np.zeros((constraints,variables))
        A[0] = self.expected_returns # Sum of weights = 1
        A[1:variables+1] = np.eye(variables) - np.ones((variables,variables))*(-self.max_leverage)
        A[-variables:] = np.eye(variables) - np.ones((variables,variables))*self.max_allocation
        
        b = np.array([1]+[0]*(constraints-1))
        sense = np.array(["="]+[">"]*variables+["<"]*variables)

        # print_equations([],A,sense,b)
        maxSharpe_model.addMConstrs(A, y, sense, b)

        # Optimize Model
        maxSharpe_model.Params.OutputFlag = 0
        maxSharpe_model.optimize()

        try:
            optimal_values = y.x
            self.weights = np.round(optimal_values/optimal_values.sum(), 2)
        except:
            print("Failed to find Optimal Weights")
        return

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        if self.tics is None:
            self.tics = price.index.tolist()
        self.calculate_inputs(date)
        self.maximise_sharpe()
        weights = pd.Series(self.weights, index = price.index)*investment#.apply(math.floor)
        weights = weights.divide(price["close"])
        return weights

    def get_weights(self)->np.array:
        return self.weights