from binance.client import Client
import pandas as pd
from stocks_env import StocksEnv
from stable_baselines3 import PPO
import os
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import pandas_ta as ta
from pon import api,secret
import warnings
warnings.filterwarnings("ignore")


def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:,'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:,['SMA3_pct','SMA6_pct','SMA9_pct','SMA12_pct','SMA25_pct','SMA50_pct','SMA200_pct','RSX']].to_numpy()[start:end]
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    _process_data = add_signals
    
def SMA(df:pd.DataFrame) -> pd.DataFrame:    
    df['SMA3'] = df['Close'].rolling(window=3).mean()
    # Calculate the percent change of the SMA
    df['SMA3_pct'] = df['SMA3'].pct_change()*1000
    
    df['SMA6'] = df['Close'].rolling(window=6).mean()
    # Calculate the percent change of the SMA
    df['SMA6_pct'] = df['SMA6'].pct_change()*1000
    
    df['SMA9'] = df['Close'].rolling(window=9).mean()
    # Calculate the percent change of the SMA
    df['SMA9_pct'] = df['SMA9'].pct_change()*1000
    
    df['SMA12'] = df['Close'].rolling(window=12).mean()
    # Calculate the percent change of the SMA
    df['SMA12_pct'] = df['SMA12'].pct_change()*1000
    
    df['SMA25'] = df['Close'].rolling(window=25).mean()
    # Calculate the percent change of the SMA
    df['SMA25_pct'] = df['SMA25'].pct_change()*1000
    
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    # Calculate the percent change of the SMA
    df['SMA50_pct'] = df['SMA50'].pct_change()*1000
    
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    # Calculate the percent change of the SMA
    df['SMA200_pct'] = df['SMA200'].pct_change()*1000
    


traindf = pd.read_csv('traindf.csv')
testdf = pd.read_csv('testdf.csv')
enddf = pd.read_csv('enddf.csv')
traindf['date'] = pd.to_datetime(traindf['date'])
testdf['date'] = pd.to_datetime(testdf['date'])
enddf['date'] = pd.to_datetime(enddf['date'])
traindf.set_index('date',inplace=True)
testdf.set_index('date',inplace=True)
enddf.set_index('date',inplace=True)


traindf['RSX'] = ta.rsx(traindf['Close'],21)
testdf['RSX'] = ta.rsx(testdf['Close'],21)
enddf['RSX'] = ta.rsx(enddf['Close'],21)

# traindf['AO'] = ta.ao(traindf['High'],traindf['Low'])
# testdf['AO'] = ta.ao(testdf['High'],testdf['Low'])

# traindf['BOP'] = ta.bop(traindf['Open'],traindf['High'],traindf['Low'],traindf['Close'])
# testdf['BOP'] = ta.bop(testdf['Open'],testdf['High'],testdf['Low'],testdf['Close'])

# traindf['CMO'] = ta.cmo(traindf['Close'])
# testdf['CMO'] = ta.cmo(testdf['Close'])

# traindf['CTI'] = ta.cti(traindf['Close'])
# testdf['CTI'] = ta.cti(testdf['Close'])

# traindf['ER'] = ta.er(traindf['Close'])
# testdf['ER'] = ta.er(testdf['Close'])

SMA(traindf)
SMA(testdf)
SMA(enddf)



# traindf['hammer'] = ta.cdl_pattern(traindf['Open'],traindf['High'],traindf['Low'],traindf['Close'],name="hammer")
# traindf['shootingstar'] = ta.cdl_pattern(traindf['Open'],traindf['High'],traindf['Low'],traindf['Close'],name="shootingstar")
# testdf['hammer'] = ta.cdl_pattern(testdf['Open'],testdf['High'],testdf['Low'],testdf['Close'],name="hammer")
# testdf['shootingstar'] = ta.cdl_pattern(testdf['Open'],testdf['High'],testdf['Low'],testdf['Close'],name="shootingstar")

# traindf['invertedhammer'] = ta.cdl_pattern(traindf['Open'],traindf['High'],traindf['Low'],traindf['Close'],name="invertedhammer")
# traindf['hangingman'] = ta.cdl_pattern(traindf['Open'],traindf['High'],traindf['Low'],traindf['Close'],name="hangingman")
# testdf['invertedhammer'] = ta.cdl_pattern(testdf['Open'],testdf['High'],testdf['Low'],testdf['Close'],name="invertedhammer")
# testdf['hangingman'] = ta.cdl_pattern(testdf['Open'],testdf['High'],testdf['Low'],testdf['Close'],name="hangingman")

# traindf['gravestonedoji'] = ta.cdl_pattern(traindf['Open'],traindf['High'],traindf['Low'],traindf['Close'],name="gravestonedoji")
# traindf['dragonflydoji'] = ta.cdl_pattern(traindf['Open'],traindf['High'],traindf['Low'],traindf['Close'],name="dragonflydoji")
# testdf['gravestonedoji'] = ta.cdl_pattern(testdf['Open'],testdf['High'],testdf['Low'],testdf['Close'],name="gravestonedoji")
# testdf['dragonflydoji'] = ta.cdl_pattern(testdf['Open'],testdf['High'],testdf['Low'],testdf['Close'],name="dragonflydoji")

# traindf['marubozu'] = ta.cdl_pattern(traindf['Open'],traindf['High'],traindf['Low'],traindf['Close'],name="marubozu")
# testdf['marubozu'] = ta.cdl_pattern(testdf['Open'],testdf['High'],testdf['Low'],testdf['Close'],name="marubozu")

# traindf[['hammer', 'shootingstar']] = traindf[['hammer', 'shootingstar']].replace({0.0: 0, 100: 1,-100:-1}).astype(float)
# traindf[['invertedhammer', 'hangingman']] = traindf[['invertedhammer', 'hangingman']].replace({0.0: 0, 100: 1,-100:-1}).astype(float)
# traindf[['gravestonedoji', 'dragonflydoji']] = traindf[['gravestonedoji', 'dragonflydoji']].replace({0.0: 0, 100: 1,-100:-1}).astype(float)
# traindf[['marubozu']] = traindf[['marubozu']].replace({0.0: 0, 100: 1,-100:-1}).astype(float)

# testdf[['hammer', 'shootingstar']] = testdf[['hammer', 'shootingstar']].replace({0.0: 0, 100: 1,-100:-1}).astype(float)
# testdf[['invertedhammer', 'hangingman']] = testdf[['invertedhammer', 'hangingman']].replace({0.0: 0, 100: 1,-100:-1}).astype(float)
# testdf[['gravestonedoji', 'dragonflydoji']] = testdf[['gravestonedoji', 'dragonflydoji']].replace({0.0: 0, 100: 1,-100:-1}).astype(float)
# testdf[['marubozu']] = testdf[['marubozu']].replace({0.0: 0, 100: 1,-100:-1}).astype(float)
#Awesome Oscillator: ao,Balance of Power: bop,Chande Momentum Oscillator: cmo,Correlation Trend Indicator: cti,Efficiency Ratio: er.
#Moving Average Convergence Divergence: macd,Schaff Trend Cycle: stc, Slope: slope, Stochastic Oscillator: stoch, Stochastic RSI: stochrsi
#Trix: trix


# traindf['PCTOpen'] = traindf['Open'].pct_change()
# traindf['PCTHigh'] = traindf['High'].pct_change()
# traindf['PCTLow'] = traindf['Low'].pct_change()
# traindf['PCTClose'] = traindf['Close'].pct_change()
# traindf['PCTVolume'] = traindf['Volume'].pct_change()

# testdf['PCTOpen'] = testdf['Open'].pct_change()
# testdf['PCTHigh'] = testdf['High'].pct_change()
# testdf['PCTLow'] = testdf['Low'].pct_change()
# testdf['PCTClose'] = testdf['Close'].pct_change()
# testdf['PCTVolume'] = testdf['Volume'].pct_change()

# enddf['PCTOpen'] = enddf['Open'].pct_change()
# enddf['PCTHigh'] = enddf['High'].pct_change()
# enddf['PCTLow'] = enddf['Low'].pct_change()
# enddf['PCTClose'] = enddf['Close'].pct_change()
# enddf['PCTVolume'] = enddf['Volume'].pct_change()

#ИЗМЕНИ ИМЯ
modelname = 'test205'
log_path = os.path.join('logs')
model_path = os.path.join('models',f'{modelname}')
# stats_path = os.path.join(log_path, "vec_normalize.pkl")
window_size = 50
start_index = window_size
end_index = len(traindf)


env = MyCustomEnv(df=traindf, frame_bound=(start_index+202,end_index), window_size=window_size)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path,learning_rate=0.000002,seed=123)
# model = PPO.load("models\\PPO_NEWENV_EQREW_LR=3e-0\\1990000.zip",env=env)


TIMESTEPS = 10000
for i in range(1,401):
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name=modelname)
    model.save(os.path.join(f'{model_path}',f'{TIMESTEPS*i}'))