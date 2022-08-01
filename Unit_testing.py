from Functions_main import *

# Read in a test dataset
df = pd.read_csv('UnitTest_data.csv')


# Retrieve RSI values
rsi = get_rsi(
    price = df['Ask Price'], 
    period = 4)

### TODO: to verify whether the RSI period 4 is calculated correctly 

# Retrieve Moving Average values
ma = get_MA(
    price = df['Ask Price'],
    MA = 5
)

### TODO: to verify whether the Moving Average 5 is calculated correctly

atr = get_atr(
    high_price = df['ask price(high)'],
    low_price = df['ask price(low)'],
    close_price = df['Ask Price'],
    period = 4
)

### TODO: to verify whether ATR 4 is calculated correctly

df['RSI_ask_4'] = rsi
df['MA_ask_5'] = ma
df['ATR_4'] = atr

df.to_csv('Output_file.csv')

