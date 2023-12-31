import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from portfolio.portfolio_bkp import Portfolio

if __name__ == "__main__":
    start_date = pd.to_datetime("2010-01-01")
    end_date = pd.to_datetime("2023-04-30")
    rebalance_freq = 3
    reconstitution_freq = 12
    portfolio_tickers = ["SPY", "ACWX", "VCIT", "IGOV", "USO"]  # ["SGOV", "QCOM", "AAPL", "X", "MERC"]

    benchmark_weights = pd.Series(data=[0.30, 0.25, 0.20, 0.15, 0.10],
                                  index=portfolio_tickers)

    stocks = yf.Tickers(" ".join(portfolio_tickers))

    stock_data = stocks.download(" ".join(portfolio_tickers), start=start_date, end=end_date)[
        ["Close", "Dividends"]]  # .dropna()
    stock_data.rename({"Close": "close", "Dividends": "div"}, axis=1, inplace=True)
    stock_data["close"].plot(figsize=(30, 10))

    for tic in portfolio_tickers:
        stock_data[("shares", tic)] = pd.Series([1] * len(stock_data.index), index=stock_data.index)

    portfolio_tickers = stock_data["close"].columns

    ticker_set = dict()
    for year in range(start_date.year, end_date.year + 1):
        ticker_set[year] = portfolio_tickers

    # Benchmark Portfolio
    benchmark_portfolio = Portfolio(target="Baseline",
                                    tickerset=ticker_set, investment=100,
                                    trading_algo="constant_weight_Algo",
                                    rebalance=rebalance_freq, reconstitute=reconstitution_freq)

    benchmark_val = pd.Series([], dtype=float)
    benchmark_returns = pd.Series([], dtype=float)

    for date in stock_data.index:
        _ = stock_data.loc[date]
        _ = _.unstack().T
        valuation, returns = benchmark_portfolio.run(date=date, price=_, tickerlist=portfolio_tickers)
        if (valuation, returns) == (-1, -1):
            print(benchmark_val)
            print(benchmark_val.index[-1], date.date(), valuation, returns)
            continue
        benchmark_val[date] = valuation
        benchmark_returns[date] = returns

    benchmark_portfolio.echo()

    # Plot
    plt.figure().set_size_inches(30, 10, forward=True)

    plt.plot(benchmark_val, color="blue", alpha=0.5, label=benchmark_portfolio.target)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Portfolio Net Asset Value (in million $)")
    plt.title("Custom benchmark - Valuation")
    plt.show()
