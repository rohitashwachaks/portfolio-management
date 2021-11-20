import pandas as pd
import numpy as np

exec(open("trading_algo.py").read())

# import TradingAlgo


class Portfolio:
    def __init__(self, target: str,
                tickerset: set,
                trading_algo: TradingAlgo,
                investment: int = 1000,
                rebalance : int = 0,
                reconstitute : int = 1) -> None:
        '''
            Initialise portfolio
        '''
        self.IsWeightInitialised = False
        self.target = target
        self.ticker_set = tickerset
        self.tickers = []
        self.trading_algo = trading_algo
        
        self.start_nav = investment
        self.prevday_nav = investment
        self.current_nav = investment
        self.dividents = pd.Series([], dtype= float)                      # ignore dividents for now
        
        self.weights = pd.Series([], dtype= float)                        # Equally balanced portfolio. Should Contain number of stocks owned. Ideally, Whole Numbers
        self.last_rebalancing_date = pd.to_datetime("01-01-1990")         # First rebalancing at 01st, Jan
        self.rebalancing_interval = rebalance - 1                         # in month (plus 1, because that month included)
        self.last_reconstitution_date = pd.to_datetime("01-01-1990")      # First reconstitution at 01st, Jan
        self.reconstitution_interval = reconstitute                       # in years
        pass

    def update_tickerlist(self, date: pd.Timestamp) -> None:
        self.tickers = self.ticker_set[date.year]
        return

    def isUpdateAllowed(self, date: pd.Timestamp, mode : str = "rebalance") -> bool:
        '''
            Calculate Number of months between two dates
        '''
        diff = False
        if mode == "rebalance":
            diff = (date.year - self.last_rebalancing_date.year)*12 + date.month - self.last_rebalancing_date.month
            diff = diff >= self.rebalancing_interval
        elif mode == "reconstitute":
            diff = (date.year - self.last_reconstitution_date.year)*12 + date.month - self.last_reconstitution_date.month
            diff = diff >= self.reconstitution_interval
        return diff

    def reconstitute(self, price: pd.DataFrame, date: pd.Timestamp, tickerlist: list) -> None:
        '''
            Function to Reconstitute our portfolio.
            Checks:
                - weights updated only after "Reconstitution" period has passed
        '''
        print("Reconstitution on: "+date.strftime("%d-%b-%Y"))

        if self.isUpdateAllowed(date= date, mode="reconstitute"):
            # Convert weights to number of shares in portfolio
            # Insert reconstitution code here
            if self.last_reconstitution_date != pd.to_datetime("01-01-1990"):
                self.current_nav = self.weights.multiply(price["close"]).sum() + self.dividents.sum()
                self.prevday_nav = self.current_nav
            
            self.update_tickerlist(date)
            self.dividents = pd.Series([0]*len(self.tickers), index=self.tickers)
            self.weights = pd.Series([1]*len(self.tickers), index=self.tickers)

            self.IsWeightInitialised = False

            price = price[price.apply(lambda x: (x.name in self.tickers),axis = 1)]

            if price.shape[0] == 0:
                return

            self.rebalance(price= price, date= date)
        else:
            raise Exception("Reconstitution Interval not passed.\n\t\
                            Last Reconstituted on: {last_recons}\n\t\
                            Requesting Reconstitution on: {curr_recons}".format(last_recons = self.last_reconstitution_date.strftime("%d-%b-%Y"),
                                                                        curr_recons = date.strftime("%d-%b-%Y")))
        self.last_reconstitution_date = date
        return

    def rebalance(self, price, date: pd.Timestamp = None) -> None:
        '''
            Function to update the Number of shares held by our portfolio.
            Checks:
                - Do ticker lists match?
                - weights updated only after "rebalance" period has passed
        '''
        print("\tRebalancing on: "+date.strftime("%d-%b-%Y"))
        
        if self.isUpdateAllowed(date= date, mode="rebalance"):
            difference_set = {x for x in price.index if x not in set(self.tickers)}
            if len(difference_set) == 0:

                if self.IsWeightInitialised:
                    self.dividents = pd.Series([0.0]*len(self.tickers), index=self.tickers)
                    self.prevday_nav = self.current_nav
                    self.current_nav = self.weights.multiply(price["close"]).sum() + self.dividents.sum()
                    
                
                self.weights = self.trading_algo.run(price, self.current_nav)
                self.last_rebalancing_date = date
                self.IsWeightInitialised = True
            else:
                raise Exception(" Input and Portfolio Tickers Not Matching. Net mismatch: {mismatch}\n".format(mismatch = len(difference_set))+", ".join(list(difference_set))); 
        else:
            raise Exception("rebalance Interval not passed.\n\t\
                            Last rebalanced on: {last_rebal}\n\t\
                            Requesting rebalance on: {curr_rebal}".format(last_rebal = self.last_rebalancing_date.strftime("%d-%b-%Y"),
                                                                        curr_rebal = date.strftime("%d-%b-%Y")))
        return

    def run(self, price: pd.DataFrame, date: pd.Timestamp, tickerlist:list = None) -> tuple:
        '''
            This function will be run on every date
        '''
        price.fillna(0, inplace = True)
        self.dividents += price["div"]
        
        if sum(price.shares) != 0:
            if self.isUpdateAllowed(date,mode="reconstitute"):
                self.reconstitute(date= date, price= price, tickerlist= tickerlist)
            else: 
                price = price[price.apply(lambda x: (x.name in self.tickers),axis = 1)]
                if price.shape[0] == 0:
                    return (-1,-1)
                if self.isUpdateAllowed(date):
                    self.rebalance(date= date, price = price)
                else:
                    self.prevday_nav = self.current_nav
                    self.current_nav = self.weights.multiply(price["close"]).sum() + self.dividents.sum()
        
        return self.current_nav, self.get_returns()

    def get_nav(self, price: pd.DataFrame) -> float:
        nav = self.weights.multiply(price["close"]).sum() + self.dividents.sum()
        return nav

    def get_returns(self) -> float:
        '''
            Function to calculate returns of our portfolio
        '''
        return ((self.current_nav - self.prevday_nav + self.dividents.sum())/(self.prevday_nav))

    def echo(self) -> None:
        print("Target Index:",self.target)
        print("Trading Algorithm:",self.trading_algo.identifier)
        print("Starting Investment:",self.start_nav)
        print("Current Investment:",self.current_nav)
        if len(self.tickers) > 0:
            print("Constituent Indices:", ", ".join(self.tickers[0:5]+["\b\b ..."]))
        if len(self.weights) > 0:
            print("Constituent Weights:", self.weights[0:5])
        pass