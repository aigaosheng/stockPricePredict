import pyfolio as pf

# silence warnings
import warnings
warnings.filterwarnings('ignore')
import pandas_ta as ta

symbol = "NVDA"
df = pd.DataFrame()
df = df.ta.ticker(symbol)

stock_rets = df["Close"].pct_change().dropna()#.tail(252)
stock_rets.name = symbol


# stock_rets = pf.utils.get_symbol_rets(symbol)
stock_rets.index = stock_rets.index.map(lambda x: x.strftime("%Y-%m-%d"))
stock_rets.index = stock_rets.index.map(lambda x: datetime.strptime(x, "%Y-%m-%d"))

pf.create_returns_tear_sheet(stock_rets, live_start_date='2015-12-1')