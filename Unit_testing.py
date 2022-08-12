from ast import parse
import os 
os.chdir('/Users/ignatiusgoh/Desktop/Grasshopper/Unit Testing')
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
    price = ((df['Ask Price'] + df['Bid Price'])/2),
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
df['MA_5'] = ma
df['ATR_4'] = atr
df = df.loc[~(df == 0).all(axis = 1)]
df = df.dropna()
# df.to_csv('Output_file.csv')

# Preprocessing should output a csv file with the name of the stock and the year input with the parameters of rsi periods, 
# atr periods and moving average calculated 
df = pd.read_csv('UnitTest_data.csv')
rsi_period = [4]
atr_period = [4]
moving_average = [5,6]
preprocessing(df,2020, rsi_period, atr_period, moving_average)

### TODO: to verify whether calculated values in Output file and calculated values in STOCK|XTKS|8473_2020.csv are the same 


# Initialize set of rsi periods, oversold and overbought values to enter/exit a trade
# What should the user see? 
# If the current rsi value is below the oversold values and the previous rsi value is above, it will return a 1 
# if current rsi value is above the overbought value and previous rsi value is below it will return a -1

df = pd.read_csv('STOCK|XTKS|8473|_2020.csv', parse_dates = ['time','Date'])
rsi_periods = [4] # This needs to be consistent with the dataset 
rsi_oversold = [30,40] # List of oversold values we want to test from 
rsi_overbought = 60 

long_df = to_enter_long(df,rsi_periods,rsi_oversold,rsi_overbought)

### TODO: to verify whether long entries and exits based on RSI values executed correctly


# Similar to long entries 
# If the current rsi value is above the overbought values and the previous rsi value is below, it will return a 1 
# if current rsi value is below the oversold value and previous rsi value is above it will return a -1
rsi_periods = [4]
rsi_overbought = [60,70]
rsi_oversold = 40

short_df = to_enter_short(df,rsi_periods,rsi_overbought,rsi_oversold)

### TODO: to verify whether short entries and exits based on RSI values executed correctly

# Testing of exit conditions 
# We will test based on rsi period 4 and oversold threshold of 30 
# We should receive a data frame of all the trades that were conducted and the reason why it closed the position

rsi_list = list(long_df['RSI_signal_long_4_30'])
trades_main = pd.DataFrame()
for index, value in enumerate(rsi_list):
    if value == 0: continue
    elif value == -1: continue 
    elif value == 1: 
        if df['Hour'][index] != 5 and df['Hour'][index] != 6: # No entry one hour before market closes 
            remaining_rsi = rsi_list[index+1:]
            remaining_df = long_df.iloc[index: , :].reset_index(drop = True)
            trade = long_position(remaining_df, remaining_rsi, atr_exit_multiple = 0.04, atr_stoploss_multiple = 0.6)
            trades_main = pd.concat([trades_main,trade])

trades_main

### TODO: based on the list dataframe with signals of long entries and exits identified, manually look for the exit conditions as stated in the trades main file 
# It is also important to note the exit prices especially for stop-losses and exit prices because the tester will need to reference the ATR value at the point and calculate the stop-loss and exit prices

# Similar concept is utilied to test the exit condition of the short trades 
rsi_list = list(short_df['RSI_signal_long_4_30'])
trades_main = pd.DataFrame()
for index, value in enumerate(rsi_list):
    if value == 0: continue
    elif value == -1: continue 
    elif value == 1: 
        if df['Hour'][index] != 5 and df['Hour'][index] != 6: # No entry one hour before market closes 
            remaining_rsi = rsi_list[index+1:]
            remaining_df = short_df.iloc[index: , :].reset_index(drop = True)
            trade = short_position(remaining_df, remaining_rsi, atr_exit_multiple = 0.04, atr_stoploss_multiple = 0.6)
            trades_main = pd.concat([trades_main,trade])


### TODO: based on the list dataframe with signals of long entries and exits identified, manually look for the exit conditions as stated in the trades main file 
# It is also important to note the exit prices especially for stop-losses and exit prices because the tester will need to reference the ATR value at the point and calculate the stop-loss and exit prices
