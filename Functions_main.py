import matplotlib.pyplot as plt
import datetime
import pandas as pd 
import numpy as np 
import math
from mpl_toolkits import mplot3d
from tqdm.notebook import trange, tqdm
import warnings
warnings.filterwarnings("ignore")
import pandas as p
# import dask
# import graphviz
from multiprocessing import get_context
import threading

def get_rsi(
    price,
    period : int
    ):
    
    ret = price.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = up_series.ewm(com = period - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = period - 1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_MA(
    price, 
    MA: int
    ):

    ma = price.rolling(MA).mean()

    return ma

def get_atr(
    high_price,
    low_price,
    close_price,
    period: int
    ): 
    # high_low = data['ask price(high)'] - data['ask price(low)']
    # high_close = np.abs(data['ask price(high)'] - data['Ask Price'].shift())
    # low_close = np.abs(data['ask price(low)'] - data['Ask Price'].shift())
    high_low = high_price - low_price
    high_close = np.abs(high_price - close_price.shift())
    low_close = np.abs(low_price - close_price.shift())

    ranges = pd.concat([high_low,high_close,low_close], axis = 1)
    true_range = np.max(ranges, axis = 1)
    atr = true_range.rolling(14).sum()/period

    return atr

def preprocessing(
    df, 
    date : str, 
    rsi_period : list, 
    atr_period : list, 
    moving_average: list
    ): 
    tickerlist = list(df['sym'].unique())
    
    for ticker in tickerlist:
        temp_df = df[df['sym'] == ticker]
        temp_df = temp_df.sort_values(by = 'time')
        temp_df = temp_df.reset_index(drop = True)
        
        for period in rsi_period: 
            temp_df[f'RSI_ask_{period}'] = get_rsi(temp_df['Ask Price'], period)
            temp_df[f'RSI_ask_{period}_shift'] = temp_df['RSI_ask_{}'.format(period)].shift(1)
            temp_df[f'RSI_bid_{period}'] = get_rsi(temp_df['Bid Price'], period)
            temp_df[f'RSI_bid_{period}_shift'] = temp_df['RSI_bid_{}'.format(period)].shift(1)

#         for ma in moving_average: 
#             temp_df = get_MA(temp_df,ma)
#             # temp_df['MA_{}_shift'.format(ma)] = temp_df['MA_{}'.format(ma)].shift(1)
        
        # for ma in moving_average: 
        #     temp_df[f'MA_{ma}'] = get_MA(
        #         ((temp_df['Bid Price'] + temp_df['Ask Price'])/2), 
        #         ma
        #         )

#         # ma_crosses = []
#         for ma in moving_average:
#             moving_average_upper = moving_average[1:]
#             for ma_upper in moving_average_upper:
#                 if ma < ma_upper:
#                     temp_df['MA_{}_{}_diff'.format(ma,ma_upper)] = abs(temp_df['MA_{}'.format(ma)] - temp_df['MA_{}'.format(ma_upper)])
#                     # ma_crosses.append('{}_{}'.format(ma,ma_upper))
#             moving_average_upper = moving_average_upper[1:]

        for atr in atr_period:
            temp_df['ATR'] = get_atr(
                high_price = temp_df['ask price(high)'],
                low_price = temp_df['ask price(low)'],
                close_price = temp_df['Ask Price'],
                period = atr
            )
        
        temp_df = temp_df.loc[~(temp_df == 0).all(axis = 1)]
        temp_df = temp_df.dropna()
        temp_df = temp_df.iloc[1: , :]
        temp_df['time'] = pd.to_datetime(temp_df['time'])
        temp_df['Hour'] = temp_df['time'].dt.hour
        temp_df.to_csv('{ticker_name}_{date_input}.csv'.format(ticker_name = ticker, date_input = date))


def to_enter_long(
    df,
    rsi_periods : list, 
    rsi_oversold : list, 
    rsi_overbought : int
    ):
    for period in rsi_periods:
        for low in rsi_oversold: 
            counter = 0
            rsi_signal = []
            while counter < len(df['RSI_ask_{}'.format(period)]):
                
                # if current rsi is below oversold and previous value is above --> signifying long position entry 
                if df['RSI_ask_{}'.format(period)][counter] <= low and df['RSI_ask_{}_shift'.format(period)][counter] > low: 
                    rsi_signal.append(1)
                
                # If current rsi is above overbought and previous value is below --> exit long position 
                elif df['RSI_ask_{}'.format(period)][counter] >= rsi_overbought and df['RSI_ask_{}_shift'.format(period)][counter] < rsi_overbought:
                    rsi_signal.append(-1)
                
                else: 
                    rsi_signal.append(0)
                counter += 1 
            
            df['RSI_signal_long_{rsi_period}_{oversold_threshold}'.format(rsi_period = period, oversold_threshold = low)] = rsi_signal 

    return df

def to_enter_short(
    df,
    rsi_periods : list, 
    rsi_overbought : list, 
    rsi_oversold : int
    ):
    for period in rsi_periods:
        for high in rsi_overbought: 
            counter = 0
            rsi_signal = []
            while counter < len(df['RSI_ask_{}'.format(period)]):
                
                # if current rsi is above overbought and previous value is below --> signifying short position entry 
                if df['RSI_ask_{}'.format(period)][counter] >= high and df['RSI_ask_{}_shift'.format(period)][counter] < high: 
                    rsi_signal.append(1)
                
                # If current rsi is below oversold and previous value is up --> exit short position 
                elif df['RSI_ask_{}'.format(period)][counter] <= rsi_oversold and df['RSI_ask_{}_shift'.format(period)][counter] > rsi_oversold:
                    rsi_signal.append(-1)
                
                else: 
                    rsi_signal.append(0)
                counter += 1 
            
            df['RSI_signal_short_{rsi_period}_{overbought_threshold}'.format(rsi_period = period, overbought_threshold = high)] = rsi_signal 

    return df

def long_position(
    df,
    remaining_list : list, 
    atr_exit_multiple : int, 
    atr_stoploss_multiple : int
    ): 
    
    # Preset: the first row of this new dataframe is the price of which we entered the position with its respective technical values
    entry_price = df['Ask Price'][0]
    atr = df['ATR'][0]
    exit_price = entry_price + (atr * atr_exit_multiple)
    stoploss_price = entry_price - (atr * atr_stoploss_multiple)
    entry_time = df['time'][0]

    # Dropping the first row of the dataframe 
    df = df.iloc[1: , :].reset_index(drop = True)

    for index, rsi in enumerate(remaining_list): 
        current_time = df['time'][index]
        current_price = df['Bid Price'][index]
        if rsi == 0 or rsi == 1: 
            from datetime import timedelta 
            if df['Hour'][index] == 6: #Closing the position if it is the end of day 
                close_price = current_price
                reason = 'EOD Close'
                break 

            elif (current_time - entry_time) >= timedelta(hours = 4): #Closing the position if the position is held for more than 4 hours 
                close_price = current_price
                reason = '4 Hour Close'
                break
            
            elif current_price >= exit_price: #Closing the position if current price is >= exit price 
                close_price = current_price
                reason = 'Exit Profits Close'
                break

            elif current_price <= stoploss_price: #Closing the position if the current price is <= stoploss price
                close_price = current_price
                reason = 'Stoploss Close'
                break
                
            elif index + 1 == len(remaining_list):
                close_price = current_price
                reason = 'End of list'
                break

            else: 
                continue 
            
        elif rsi == -1: #Closing the position if the RSI overbought signal has been hit
            close_price = current_price
            reason = 'RSI Overbought Close'
            break 
            
        # else: 
        #     close_price = current_price
        #     reason = 'End of list'
        #     break 

    data = {
        'Ticker' : df['sym'][0],
        'Entry_Time' : entry_time,
        'Entry_Price' : entry_price,
        'Exit_Time' : current_time,
        'Exit_Price' : close_price,
        'Exit_Reason' : reason,
        'Trade_Type' : 'Long'
    }

    trades_df = pd.DataFrame(data, index = [0])
    
    return trades_df 

# Create function to determine exit positions (short)
def short_position(
    df,
    remaining_list : list, 
    atr_exit_multiple : int, 
    atr_stoploss_multiple : int
    ): 
    
    # Preset: the first row of this new dataframe is the price of which we entered the position with its respective technical values
    entry_price = df['Ask Price'][0]
    atr = df['ATR'][0]
    exit_price = entry_price - (atr * atr_exit_multiple)
    stoploss_price = entry_price + (atr * atr_stoploss_multiple)
    entry_time = df['time'][0]

    # Dropping the first row of the dataframe 
    df = df.iloc[1: , :].reset_index(drop = True)

    for index, rsi in enumerate(remaining_list): 
        
        current_time = df['time'][index]
        current_price = df['Bid Price'][index]
        if rsi == 0 or rsi == 1: 
            from datetime import timedelta 
            if df['Hour'][index] == 6: #Closing the position if it is the end of day 
                close_price = current_price
                reason = 'EOD Close'
                break 

            elif (current_time - entry_time) >= timedelta(hours = 4): #Closing the position if the position is held for more than 4 hours 
                close_price = current_price
                reason = '4 Hour Close'
                break

            elif current_price <= exit_price: #Closing the position if current price is >= exit price 
                close_price = current_price
                reason = 'Exit Profits Close'
                break

            elif current_price >= stoploss_price: #Closing the position if the current price is <= stoploss price
                close_price = current_price
                reason = 'Stoploss Close'
                break
                
            elif index + 1 == len(remaining_list):
                close_price = current_price
                reason = 'End of list'
                break
            
            else: 
                continue 
            
        elif rsi == -1: #Closing the position if the RSI overbought signal has been hit
            close_price = current_price
            reason = 'RSI Oversold Close'
            break 
            
        # else: 
        #     close_price = current_price
        #     reason = 'End of list'
        #     break 

    data = {
        'Ticker' : df['sym'][0],
        'Entry_Time' : entry_time,
        'Entry_Price' : entry_price,
        'Exit_Time' : current_time,
        'Exit_Price' : close_price,
        'Exit_Reason' : reason,
        'Trade_Type' : 'Short'
    }

    trades_df = pd.DataFrame(data, index = [0])
    
    return trades_df 

def long_rsi_parameter_extraction(
    df, 
    rsi_periods : list, 
    rsi_oversold_long : list, 
    atr_exit_multiple : int, 
    atr_stoploss_multiple : int
    ):
    main_dict = {
        'Ticker' : df['sym'][0],
        'RSI_Period' : [],
        'RSI_Oversold_Threshold' : [],
        'Average_PnL' : [],
        'Type' : 'Long'
        
    }
    for rsi in rsi_periods:
        for low in rsi_oversold_long:
            trades_main = pd.DataFrame()
            rsi_list = list(df['RSI_signal_long_{}_{}'.format(rsi,low)])
            # Loop through rsi values to enter positions 
            for index, value in enumerate(rsi_list):
                if value == 0: continue
                elif value == -1: continue 
                elif value == 1: 
                    if df['Hour'][index] != 5 and df['Hour'][index] != 6: # No entry one hour before market closes 
                        remaining_rsi = rsi_list[index+1:]
                        remaining_df = df.iloc[index: , :].reset_index(drop = True)
                        trade = long_position(remaining_df, remaining_rsi, atr_exit_multiple, atr_stoploss_multiple)
                        trades_main = pd.concat([trades_main, trade])
            
            if trades_main.empty == False:
                trades_main = trades_main.reset_index(drop = True)           

                # Extract the PnL analytics of all trades 
                trades_main['Change'] = trades_main['Exit_Price'] - trades_main['Entry_Price'] # Nominal PnL 
                trades_main['Percentage_Change'] = ((trades_main['Exit_Price'] - trades_main['Entry_Price']) / trades_main['Entry_Price'])*100 # Percentage PnL

                # Append values to main_dict
                main_dict['RSI_Period'].append(rsi)
                main_dict['RSI_Oversold_Threshold'].append(low)
                main_dict['Average_PnL'].append(trades_main['Percentage_Change'].mean())

    analytics = pd.DataFrame(main_dict)
    analytics = analytics.sort_values(by = 'Average_PnL', ascending = False)
    analytics = analytics.reset_index(drop = True)

    return analytics

def short_rsi_parameter_extraction(
    df, 
    rsi_periods : list, 
    rsi_overbought_short : list, 
    atr_exit_multiple : int, 
    atr_stoploss_multiple : int
    ):
    main_dict = {
        'Ticker' : df['sym'][0],
        'RSI_Period' : [],
        'RSI_Overbought_Threshold' : [],
        'Average_PnL' : [],
        'Type': 'Short'
    }
    for rsi in rsi_periods:
        for high in rsi_overbought_short:
            trades_main = pd.DataFrame()
            rsi_list = list(df['RSI_signal_short_{}_{}'.format(rsi,high)])
            # Loop through rsi values to enter positions 
            for index, value in enumerate(rsi_list):
                if value == 0: continue
                elif value == -1: continue 
                elif value == 1: 
                    if df['Hour'][index] != 5 and df['Hour'][index] != 6: # No entry one hour before market closes 
                        remaining_rsi = rsi_list[index+1:]
                        remaining_df = df.iloc[index: , :].reset_index(drop = True)
                        trade = short_position(remaining_df, remaining_rsi, atr_exit_multiple, atr_stoploss_multiple)
                        trades_main = pd.concat([trades_main, trade])
            
            if trades_main.empty == False:
                trades_main = trades_main.reset_index(drop = True)           

                # Extract the PnL analytics of all trades 
                trades_main['Change'] = trades_main['Entry_Price'] - trades_main['Exit_Price'] # Nominal PnL 
                trades_main['Percentage_Change'] = ((trades_main['Entry_Price'] - trades_main['Exit_Price']) / trades_main['Entry_Price'])*100 # Percentage PnL

                # Append values to main_dict
                main_dict['RSI_Period'].append(rsi)
                main_dict['RSI_Overbought_Threshold'].append(high)
                main_dict['Average_PnL'].append(trades_main['Percentage_Change'].mean())

    analytics = pd.DataFrame(main_dict)
    analytics = analytics.sort_values(by = 'Average_PnL', ascending = False)
    analytics = analytics.reset_index(drop = True)

    return analytics

def trade_long(
    df, 
    rsi_period : int, 
    rsi_oversold : int, 
    atr_exit_multiple : int, 
    atr_stoploss_multiple : int
    ): 
    trades_main = pd.DataFrame()
    rsi_list = list(df['RSI_signal_long_{}_{}'.format(rsi_period, rsi_oversold)])
    
    for index, value in enumerate(rsi_list):
        if value == 0: continue
        elif value == -1: continue 
        elif value == 1: 
            if df['Hour'][index] != 5 and df['Hour'][index] != 6: # No entry one hour before market closes 
                remaining_rsi = rsi_list[index+1:]
                remaining_df = df.iloc[index: , :].reset_index(drop = True)
                trade = long_position(remaining_df, remaining_rsi, atr_exit_multiple, atr_stoploss_multiple)
                trades_main = pd.concat([trades_main, trade])
    
    return trades_main

def trade_short(
    df, 
    rsi_period : int, 
    rsi_overbought : int, 
    atr_exit_multiple : int, 
    atr_stoploss_multiple : int
    ): 
    trades_main = pd.DataFrame()
    rsi_list = list(df['RSI_signal_short_{}_{}'.format(rsi_period, rsi_overbought)])
    
    for index, value in enumerate(rsi_list):
        if value == 0: continue
        elif value == -1: continue 
        elif value == 1: 
            if df['Hour'][index] != 5 and df['Hour'][index] != 6: # No entry one hour before market closes 
                remaining_rsi = rsi_list[index+1:]
                remaining_df = df.iloc[index: , :].reset_index(drop = True)
                trade = short_position(remaining_df, remaining_rsi, atr_exit_multiple, atr_stoploss_multiple)
                trades_main = pd.concat([trades_main, trade])
    
    return trades_main

def backtest_2020_long(df, input_dict):
    
    rsi_periods = input_dict['rsi_periods']
    rsi_oversold_long = input_dict['rsi_oversold_long']
    rsi_overbought_long = input_dict['rsi_overbought_long']
    atr_exit_multiple = input_dict['atr_exit_multiple']
    atr_stoploss_multiple = input_dict['atr_stoploss_multiple']
    backtesting_duration = input_dict['backtesting_duration']
    trading_duration = input_dict['trading_duration']

    df = to_enter_long(df, rsi_periods, rsi_oversold_long, rsi_overbought_long)
    # df = to_enter_short(df, rsi_periods, rsi_overbought_short, rsi_oversold_short)
    
    duration_dict = {
        'Backtesting_Duration' : [],
        'Trading_Duration' : [],
        'Average_PnL' : [],
        'Ticker' : df['sym'][0]
    }

    datelist = list(df['Date'].unique())

    for b_duration in backtesting_duration: 
        # Initialize start counter & end counter of dates to be backtesting on 
        start_counter_init = 0 
        end_counter_init = 0 
        
        # Add element backtesting duration to the counter 
        end_counter_init += b_duration

        # Extracting data of the stock between the start of backtesting date and end 
        backtesting_dates = datelist[start_counter_init : end_counter_init]
        backtesting_filter = df['Date'].isin(backtesting_dates)
        backtesting_df = df[backtesting_filter]
        backtesting_df = backtesting_df.reset_index(drop = True)

        # Extract results from the different combinations 
        df_analytics_long = long_rsi_parameter_extraction(backtesting_df, rsi_periods, rsi_oversold_long, atr_exit_multiple, atr_stoploss_multiple)
        # df_analytics_short = short_rsi_parameter_extraction(backtesting_df, rsi_periods, rsi_overbought_short, atr_exit_multiple, atr_stoploss_multiple)

        for t_duration in trading_duration:

            trade_results_main = pd.DataFrame()
            start_counter = end_counter_init
            end_counter = start_counter 

            while end_counter < len(datelist): 
                
                # Extraction data of the stock between start of trading duration and end of trading duration 
                # The end of trading duration will be appended by the element in trading duration list
                end_counter += t_duration 
                trading_dates = datelist[start_counter : end_counter]
                trading_filter = df['Date'].isin(trading_dates)
                trading_df = df[trading_filter]
                
                # Execute trades based on optimal parameters identified
                trading_df = trading_df.reset_index(drop = True)
                if df_analytics_long.empty == False:
                    long_trades = trade_long(trading_df, df_analytics_long['RSI_Period'][0], df_analytics_long['RSI_Oversold_Threshold'][0], atr_exit_multiple, atr_stoploss_multiple)
                    trade_results_main = pd.concat([trade_results_main, long_trades])
                
                # if df_analytics_short.empty == False:
                #     short_trades = trade_short(trading_df, df_analytics_short['RSI_Period'][0], df_analytics_short['RSI_Overbought_Threshold'][0], atr_exit_multiple, atr_stoploss_multiple)
                #     trade_results_main = pd.concat([trade_results_main, short_trades])

                start_counter = end_counter 

                backtest_start_counter = start_counter - b_duration 

                # Extraction of new backtesting dataframe of the stock 
                backtest_dates = datelist[backtest_start_counter : start_counter]
                backtest_filter = df['Date'].isin(backtest_dates)
                backtesting_df = df[backtest_filter]
                backtesting_df = backtesting_df.reset_index(drop = True)

                if backtesting_df.empty == False:
                    df_analytics_long = long_rsi_parameter_extraction(backtesting_df, rsi_periods, rsi_oversold_long, atr_exit_multiple, atr_stoploss_multiple)
                    # df_analytics_short = short_rsi_parameter_extraction(backtesting_df, rsi_periods, rsi_overbought_short, atr_exit_multiple, atr_stoploss_multiple)

            if trade_results_main.empty == False:
                
                #For this set of backtesting duration & trading duration extract the PnL analytics
                trade_results_main['Change'] = trade_results_main['Exit_Price'] - trade_results_main['Entry_Price'] # Nominal PnL 
                trade_results_main['Percentage_Change'] = ((trade_results_main['Exit_Price'] - trade_results_main['Entry_Price']) / trade_results_main['Entry_Price'])*100 # Percentage PnL

                #Append values to duration_dict:
                duration_dict['Backtesting_Duration'].append(b_duration)
                duration_dict['Trading_Duration'].append(t_duration)
                duration_dict['Average_PnL'].append(trade_results_main['Percentage_Change'].mean())

    results_main = pd.DataFrame(duration_dict)

    return results_main

def backtest_2020_short(df, input_dict):

    rsi_periods = input_dict['rsi_periods']
    rsi_oversold_short = input_dict['rsi_oversold_short']
    rsi_overbought_short = input_dict['rsi_overbought_short']
    atr_exit_multiple = input_dict['atr_exit_multiple']
    atr_stoploss_multiple = input_dict['atr_stoploss_multiple']
    backtesting_duration = input_dict['backtesting_duration']
    trading_duration = input_dict['trading_duration']

    # df = to_enter_long(df, rsi_periods, rsi_oversold_long, rsi_overbought_long)
    df = to_enter_short(df, rsi_periods, rsi_overbought_short, rsi_oversold_short)
    
    duration_dict = {
        'Backtesting_Duration' : [],
        'Trading_Duration' : [],
        'Average_PnL' : [],
        'Ticker' : df['sym'][0]
    }

    datelist = list(df['Date'].unique())

    for b_duration in backtesting_duration: 
        # Initialize start counter & end counter of dates to be backtesting on 
        start_counter_init = 0 
        end_counter_init = 0 
        
        # Add element backtesting duration to the counter 
        end_counter_init += b_duration

        # Extracting data of the stock between the start of backtesting date and end 
        backtesting_dates = datelist[start_counter_init : end_counter_init]
        backtesting_filter = df['Date'].isin(backtesting_dates)
        backtesting_df = df[backtesting_filter]
        backtesting_df = backtesting_df.reset_index(drop = True)

        # Extract results from the different combinations 
        # df_analytics_long = long_rsi_parameter_extraction(backtesting_df, rsi_periods, rsi_oversold_long, atr_exit_multiple, atr_stoploss_multiple)
        df_analytics_short = short_rsi_parameter_extraction(backtesting_df, rsi_periods, rsi_overbought_short, atr_exit_multiple, atr_stoploss_multiple)

        for t_duration in trading_duration:

            trade_results_main = pd.DataFrame()
            start_counter = end_counter_init
            end_counter = start_counter 

            while end_counter < len(datelist): 
                
                # Extraction data of the stock between start of trading duration and end of trading duration 
                # The end of trading duration will be appended by the element in trading duration list
                end_counter += t_duration 
                trading_dates = datelist[start_counter : end_counter]
                trading_filter = df['Date'].isin(trading_dates)
                trading_df = df[trading_filter]
                trading_df = trading_df.reset_index(drop = True)
                
                # Execute trades based on optimal parameters identified
                
                # if df_analytics_long.empty == False:
                #     long_trades = trade_long(trading_df, df_analytics_long['RSI_Period'][0], df_analytics_long['RSI_Oversold_Threshold'][0], atr_exit_multiple, atr_stoploss_multiple)
                #     trade_results_main = pd.concat([trade_results_main, long_trades])
                
                if df_analytics_short.empty == False:
                    short_trades = trade_short(trading_df, df_analytics_short['RSI_Period'][0], df_analytics_short['RSI_Overbought_Threshold'][0], atr_exit_multiple, atr_stoploss_multiple)
                    trade_results_main = pd.concat([trade_results_main, short_trades])

                start_counter = end_counter 

                backtest_start_counter = start_counter - b_duration 

                # Extraction of new backtesting dataframe of the stock 
                backtest_dates = datelist[backtest_start_counter : start_counter]
                backtest_filter = df['Date'].isin(backtest_dates)
                backtesting_df = df[backtest_filter]
                backtesting_df = backtesting_df.reset_index(drop = True)

                if backtesting_df.empty == False:
                    # df_analytics_long = long_rsi_parameter_extraction(backtesting_df, rsi_periods, rsi_oversold_long, atr_exit_multiple, atr_stoploss_multiple)
                    df_analytics_short = short_rsi_parameter_extraction(backtesting_df, rsi_periods, rsi_overbought_short, atr_exit_multiple, atr_stoploss_multiple)

            if trade_results_main.empty == False:
                
                #For this set of backtesting duration & trading duration extract the PnL analytics
                trade_results_main['Change'] = trade_results_main['Entry_Price'] - trade_results_main['Exit_Price'] # Nominal PnL 
                trade_results_main['Percentage_Change'] = ((trade_results_main['Entry_Price'] - trade_results_main['Exit_Price']) / trade_results_main['Entry_Price'])*100 # Percentage PnL

                #Append values to duration_dict:
                duration_dict['Backtesting_Duration'].append(b_duration)
                duration_dict['Trading_Duration'].append(t_duration)
                duration_dict['Average_PnL'].append(trade_results_main['Percentage_Change'].mean())

    results_main = pd.DataFrame(duration_dict)


    return results_main