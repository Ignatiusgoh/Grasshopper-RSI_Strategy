# RSI Strategy 

The RSI is an extremely useful momentum indicator to identify trend reversals in the market. 

The are 3 primary inputs: RSI period, RSI oversold, RSI overbought.

The backtesting approach looks towards firstly identifying for a particular set of stocks, how long should we be backtesting to extract the optimal RSI periods, oversold and overbought levels. Next, it identifies how long should these set of stocks be actively traded (or in other words be within our portfolio). 

At each point in time when a stock has completed it optimal trading duration, an entire backtest is conducted on the set of stocks to once again identify which stock is the most profitable to be traded next. The metrics used here to choose the stocks to replace is Sharpe Ratio.

The functions are written in a way which allows for inputs to be in different timeframes- it is dependent on the data that is fed into the functions. 

Outputs are in the format of the trades that are conducted throughout the dataset.
