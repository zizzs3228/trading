from aiogram import Bot, Dispatcher, executor, types
import schedule
import time
import asyncio
from binance.client import Client
import pandas as pd
import plotly.graph_objects as go
from pon import api,secret,BOT_API
import numpy as np
import schedule
import time
from plotly.subplots import make_subplots
import pandas_ta as ta
import threading
import warnings
warnings.filterwarnings("ignore")


bot = Bot(token=BOT_API)
dp = Dispatcher(bot)

tickers = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT','SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'LINKUSDT']


def get_price_data_binance(ticker:str,limit:int,interval = Client.KLINE_INTERVAL_1DAY)->pd.DataFrame:
    client = Client(api, secret)
    df = pd.DataFrame(client.get_klines(symbol=ticker, interval=interval, limit=limit))
    df.columns=['date','open','high','low','close','volume','close_time','d1','d2','d3','d4','d5']
    df = df.drop(['close_time','d1','d2','d3','d4','d5'],axis=1)
    df['date'] = pd.to_datetime(df['date']*1000000)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    return df
    
def get_list_of_up_imbalances(df:pd.DataFrame)->list:
    up_imbalances = []
    for i in range(2, len(df)):
        prev_candle = df.iloc[i-2]
        next_candle = df.iloc[i]
        if prev_candle['high'] < next_candle['low']:
            up_imbalances.append([round((prev_candle['high']+next_candle['low'])/2,2), prev_candle['high'],df.iloc[i-1]['date']])
    return up_imbalances

def get_list_of_down_imbalances(df:pd.DataFrame)->list:
    down_imbalances = []
    for i in range(2, len(df)):
        prev_candle = df.iloc[i-2]
        next_candle = df.iloc[i]
        if prev_candle['low'] > next_candle['high']:
            down_imbalances.append([round((prev_candle['low']+next_candle['high'])/2,2) , prev_candle['low'] , df.iloc[i-1]['date']])
    return down_imbalances

def delete_up_imbalance(up_imbalances, df:pd.DataFrame)->list:
    for i in range(0, len(df)-10):
        for imbalance in up_imbalances:
            candle = df.iloc[i]
            if candle['date']>imbalance[2]  and (candle['close'] < imbalance[0] or candle['low'] < imbalance[0]):
                up_imbalances.remove(imbalance)
    return up_imbalances[-3:]

def delete_down_imbalance(down_imbalances, df:pd.DataFrame)->list:
    for i in range(0, len(df)-10):
        for imbalance in down_imbalances:
            candle = df.iloc[i]
            if candle['date']>imbalance[2] and (candle['close'] > imbalance[0] or candle['high'] > imbalance[0]):
                down_imbalances.remove(imbalance)
    
    return down_imbalances[-3:]

def add_moving_average(period:int,df:pd.DataFrame)->pd.DataFrame:
    df['SMA'] = df['close'].rolling(period).mean()
    df['SMAslope'] = df['SMA'].pct_change()*100

    df['up_reversal'] = ((df['SMAslope'].shift(1) < 0) & (df['SMAslope'] > 0))
    df['down_reversal'] = ((df['SMAslope'].shift(1) > 0) & (df['SMAslope'] < 0)) 
    # df['charthighest'] = df['high'].rolling(6).max()[df['down_reversal'] == True]
    # df['chartlowest'] = df['low'].rolling(6).min()[df['up_reversal'] == True]
    # up_imbalances = get_list_of_up_imbalances(df)
    # down_imbalances = get_list_of_down_imbalances(df)

    # df['up_reversal'] = False
    # df['down_reversal'] = False
    # for i in range(5, len(df)-3):
    #     for imbalance in up_imbalances:
    #         if (df.iloc[i-1]['date'] < imbalance[2] < df.iloc[i+3]['date']):
    #             if (df.iloc[i-1]['SMAslope'] < 0) & (df.iloc[i]['SMAslope'] > 0):
    #                 if not (df.iloc[i-1]['up_reversal'] or df.iloc[i-2]['up_reversal'] or df.iloc[i-3]['up_reversal'] or df.iloc[i-4]['up_reversal']):
    #                     df.loc[i, 'up_reversal'] = True
    #     for imbalance in down_imbalances:
    #         if (df.iloc[i-1]['date'] < imbalance[2] < df.iloc[i+3]['date']):
    #             if ((df.iloc[i-1]['SMAslope'] > 0) & (df.iloc[i]['SMAslope'] < 0)):
    #                 if not (df.iloc[i-1]['down_reversal'] or df.iloc[i-2]['down_reversal'] or df.iloc[i-3]['down_reversal'] or df.iloc[i-4]['down_reversal']):
    #                     df.loc[i, 'down_reversal'] = True


    # df['up_reversal'] = ((df['SMAslope'].shift(1) < 0) & (df['SMAslope'] > 0))
    # df['down_reversal'] = ((df['SMAslope'].shift(1) > 0) & (df['SMAslope'] < 0))

    return df

def add_rsx_HF(period:int,df:pd.DataFrame)->pd.DataFrame:
    df["RSX"] = ta.rsx(round((df['close']+df['high']+df['low'])/3,2), period)
    df['RSXslope'] = df['RSX'].pct_change()*100
    df['RSX_up_reversal'] = ((df['RSXslope'].shift(1) < 0) & (df['RSXslope'] > 0)) & (df['RSXslope'].shift(1).rolling(5).sum()<-6)
    df['RSX_down_reversal'] = ((df['RSXslope'].shift(1) > 0) & (df['RSXslope'] < 0)) & (df['RSXslope'].shift(1).rolling(5).sum()>6)

    up_reversals = []
    down_reversals = []
    df['hidden_up_divergence'] = False
    df['hidden_down_divergence'] = False

    for i in range(0, len(df)):
        if df.iloc[i]['RSX_up_reversal']:
            up_reversals.append(min((df.iloc[i]['low'],df.iloc[i-1]['RSX'],df.iloc[i]['date']),
                (df.iloc[i-1]['low'],df.iloc[i-1]['RSX'],df.iloc[i-1]['date']),
                (df.iloc[i-2]['low'],df.iloc[i-1]['RSX'],df.iloc[i-2]['date']),
                (df.iloc[i-3]['low'],df.iloc[i-1]['RSX'],df.iloc[i-3]['date']),
                (df.iloc[i-4]['low'],df.iloc[i-1]['RSX'],df.iloc[i-4]['date']),
                (df.iloc[i-5]['low'],df.iloc[i-1]['RSX'],df.iloc[i-5]['date'])))

        if df.iloc[i]['RSX_down_reversal']:
            down_reversals.append(max((df.iloc[i]['high'],df.iloc[i-1]['RSX'],df.iloc[i]['date']),
                (df.iloc[i-1]['high'],df.iloc[i-1]['RSX'],df.iloc[i-1]['date']),
                (df.iloc[i-2]['high'],df.iloc[i-1]['RSX'],df.iloc[i-2]['date']),
                (df.iloc[i-3]['high'],df.iloc[i-1]['RSX'],df.iloc[i-3]['date']),
                (df.iloc[i-4]['high'],df.iloc[i-1]['RSX'],df.iloc[i-4]['date']),
                (df.iloc[i-5]['high'],df.iloc[i-1]['RSX'],df.iloc[i-5]['date'])))

        if (up_reversals and down_reversals):
            for up_reversal in up_reversals[-5:]:
                if (up_reversal[0] < df.iloc[i]['low']) and (up_reversal[1] > df.iloc[i]['RSX']) and up_reversal[2] < df.iloc[i]['date']:
                    df.loc[i, 'hidden_up_divergence'] = True
                
            for down_reversal in down_reversals[-5:]:
                if (down_reversal[0] > df.iloc[i]['high']) and (down_reversal[1] < df.iloc[i]['RSX']) and down_reversal[2] < df.iloc[i]['date']:
                    df.loc[i, 'hidden_down_divergence'] = True

    return df

def add_rsx_LF(period:int,df:pd.DataFrame)->pd.DataFrame:
    df["RSX"] = ta.rsx(round((df['close']+df['high']+df['low'])/3,2), period)
    df['RSXslope'] = df['RSX'].pct_change()*100
    df['RSX_up_reversal'] = ((df['RSXslope'].shift(1) < 0) & (df['RSXslope'] > 0)) & (df['RSXslope'].shift(1).rolling(5).sum()<-6)
    df['RSX_down_reversal'] = ((df['RSXslope'].shift(1) > 0) & (df['RSXslope'] < 0)) & (df['RSXslope'].shift(1).rolling(5).sum()>6)

    up_reversals = []
    down_reversals = []
    df['classic_up_divergence'] = False
    df['classic_down_divergence'] = False

    for i in range(len(df) - 200, len(df)):
        if df.iloc[i]['RSX_up_reversal']:
            up_reversals.append(min((df.iloc[i]['low'],df.iloc[i-1]['RSX'],df.iloc[i]['date']),
                (df.iloc[i-1]['low'],df.iloc[i-1]['RSX'],df.iloc[i-1]['date']),
                (df.iloc[i-2]['low'],df.iloc[i-1]['RSX'],df.iloc[i-2]['date']),
                (df.iloc[i-3]['low'],df.iloc[i-1]['RSX'],df.iloc[i-3]['date']),
                (df.iloc[i-4]['low'],df.iloc[i-1]['RSX'],df.iloc[i-4]['date']),
                (df.iloc[i-5]['low'],df.iloc[i-1]['RSX'],df.iloc[i-5]['date'])))

        if df.iloc[i]['RSX_down_reversal']:
            down_reversals.append(max((df.iloc[i]['high'],df.iloc[i-1]['RSX'],df.iloc[i]['date']),
                (df.iloc[i-1]['high'],df.iloc[i-1]['RSX'],df.iloc[i-1]['date']),
                (df.iloc[i-2]['high'],df.iloc[i-1]['RSX'],df.iloc[i-2]['date']),
                (df.iloc[i-3]['high'],df.iloc[i-1]['RSX'],df.iloc[i-3]['date']),
                (df.iloc[i-4]['high'],df.iloc[i-1]['RSX'],df.iloc[i-4]['date']),
                (df.iloc[i-5]['high'],df.iloc[i-1]['RSX'],df.iloc[i-5]['date'])))
                
        if (up_reversals and down_reversals):
            for up_reversal in up_reversals[-1:]:    
                if (up_reversal[0] > df.iloc[i]['low']) and (up_reversal[1] < df.iloc[i]['RSX']) and up_reversal[2] < df.iloc[i]['date']:
                    df.loc[i, 'classic_up_divergence'] = True

            for down_reversal in down_reversals[-1:]:       
                if (down_reversal[0] < df.iloc[i]['high']) and (down_reversal[1] > df.iloc[i]['RSX']) and down_reversal[2] < df.iloc[i]['date']:
                    df.loc[i, 'classic_down_divergence'] = True
                
    return df

async def message(message:str):
    await bot.send_message(468424685, message)
    await bot.close()

def main():
    thread = threading.Thread(target=trading)
    thread.start()

def trading():
    for ticker in tickers:
        df_higher_tf = get_price_data_binance(ticker,300,interval=Client.KLINE_INTERVAL_15MINUTE)
        df_lower_tf = get_price_data_binance(ticker,200,interval=Client.KLINE_INTERVAL_5MINUTE)
        df_lower_tf = add_rsx_LF(21,df_lower_tf)
        df_lower_tf = add_moving_average(6,df_lower_tf)
        df_higher_tf = add_rsx_HF(14,df_higher_tf)
        print(f'Работаю с {ticker}')

        if df_lower_tf.iloc[-4]['low'] > df_lower_tf.iloc[-3]['low'] and df_lower_tf.iloc[-2]['close'] > df_lower_tf.iloc[-4]['open'] and df_lower_tf.iloc[-4]['open'] > df_lower_tf.iloc[-4]['close']:
            print(f'up_reversal at {ticker}')
            if df_higher_tf.iloc[-2]['hidden_up_divergence']:
                print(f'hidden_up_divergence at {ticker}')
                if df_lower_tf.iloc[-2]['classic_up_divergence']:
                    print(f'classic_up_divergence at {ticker}')
                    stringTOsend = f"""Токен:{ticker}
Позиция: LONG
Текущая цена: {df_lower_tf.iloc[-1]['close']}
Значение RSX: {df_lower_tf.iloc[-2]['RSX']}
Цена стопа: {df_lower_tf.iloc[-3]['low']}
Время сигнала: {df_lower_tf.iloc[-1]['date']}"""
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(message(stringTOsend))

        if df_lower_tf.iloc[-4]['high'] < df_lower_tf.iloc[-3]['high'] and df_lower_tf.iloc[-2]['close'] < df_lower_tf.iloc[-4]['open'] and df_lower_tf.iloc[-4]['open'] < df_lower_tf.iloc[-4]['close']:
            print(f'down_reversal at {ticker}')
            if df_higher_tf.iloc[-2]['hidden_down_divergence']:
                print(f'hidden_down_divergence at {ticker}')
                if df_lower_tf.iloc[-2]['classic_down_divergence']:
                    print(f'classic_down_divergence at {ticker}')
                    stringTOsend = f"""Токен:{ticker}
Позиция:SHORT
Текущая цена: {df_lower_tf.iloc[-1]['close']}
Значение RSX: {df_lower_tf.iloc[-2]['RSX']}
Цена стопа: {df_lower_tf.iloc[-3]['high']}
Время сигнала: {df_lower_tf.iloc[-1]['date']}"""
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(message(stringTOsend))
        time.sleep(1)

if __name__ == '__main__':
    while True:
        current_time = time.gmtime()
        if current_time.tm_min % 5 == 0:
            break
    main()       
    schedule.every(5).minutes.do(main)

    while True:
        schedule.run_pending()