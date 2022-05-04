import talib as ta
import numpy as np

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tradingview_ta import TA_Handler, Interval, Exchange
import tensorflow as tf
from tensorflow import keras
from keras import layers
from com_var import *
import wave_process

#data to timeseries steps
def df_to_X_y2(df, window_size=6,o=1):
    print(df.shape)
    k=df.reset_index()
    ko=o
    ko=0 if o==1 else o
    l= 0 if o==1 else 3
    

    df_as_np = df.to_numpy()
    dates=[]
    X = []
    y = []
    xlast=[]
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)

        label =[df_as_np[i+window_size][ko]] if o==1 else df_as_np[i+window_size][:ko]
        y.append(label)
        dates.append(k[v][i+window_size])
        xlast.append([df_as_np[i+window_size-1][l]])
    X=np.asarray(X)
    y=np.asarray(y)
    
    print(f"hii X {X.shape}, y {y.shape}")
    return X,y,dates,xlast

#techinical analysis 
def technicalIndicators(data):
    data["MA"]=ta.MA(data['Close'], timeperiod=14, matype=0)
    
    data['macd'], data['macdsignal'], data['macdhist'] = ta.MACD(data['Close'], \
                                                                 fastperiod=7, slowperiod=14, signalperiod=5)
    data["DEMA"]=ta.DEMA(data['Close'], timeperiod=14)

    data["EMA10"]=ta.EMA(data['Close'], timeperiod=10)
    data["EMA30"]=ta.EMA(data['Close'], timeperiod=30)
    data["MOM"] = ta.MOM(data['Close'], timeperiod=14)
    
    data['ROC']=ta.ROC(data['Close'], timeperiod=14)
    data['KAMA']=ta.KAMA(data['Close'], timeperiod=14)



    return data




def getdata(ticker,period="60d",interval="15m"):
    li=["Adj Close","Volume","Open","High","Low"] if o==1 else ["Adj Close","Volume"]
    data = yf.download(tickers=ticker+"=X",period=period ,interval=interval,progress=True)
    data=data.drop(li,axis=1)
    
    data=technicalIndicators(data)
    data.dropna(how='any', inplace=True)
    data = data[data.shape[0] % batch_size:]
    validation_size1=round(data.shape[0]/6)
    test_size1=validation_size1
    test_size1

    df_train =data[:- validation_size1*2].copy(deep=True)
    df_validation = data[- validation_size1*2 :- test_size1].copy(deep=True)
    df_test = data[- test_size1 :].copy(deep=True)
    print(f'df_train.shape {df_train.shape}, df_validation.shape {df_validation.shape}, df_test.shape {df_test.shape}')
    return df_train,df_validation,df_test,data


#preprocess
def preprocess(df_train,df_test,df_validation,scaler = MinMaxScaler()):
    
    
    train_values=df_train.iloc[:,:].copy(deep=True)
    val_values=df_validation.iloc[:,:].copy(deep=True)
    test_values=df_test.iloc[:,:].copy(deep=True)
    train_values.iloc[:,:] = scaler.fit_transform(train_values)[:,:]
    val_values.iloc[:,:]=scaler.fit_transform(val_values)[:,:]
    test_values.iloc[:,:]=scaler.fit_transform(test_values)[:,:]
    X,y,dx,xtl=df_to_X_y2(train_values,window_size)
    x_val,y_val,dv,xvl=df_to_X_y2(val_values,window_size)
    x_test,y_test,dt,xtsl=df_to_X_y2(test_values,window_size)

    return X,y,x_val,y_val,x_test,y_test,train_values,val_values,test_values,scaler,dx,dv,dt,xtsl

#MODEL TRANFROMERS

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

  
def build_model(input_shape,head_size,num_heads,ff_dim,num_transformer_blocks,mlp_units,dropout=0,mlp_dropout=0,):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.LSTM(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(o)(x)
    return keras.Model(inputs, outputs)

def getrocketmodel(xtrain):
    input_shape = xtrain.shape[1:]

    model1 = build_model(input_shape,head_size=256,num_heads=4,ff_dim=4,num_transformer_blocks=4,mlp_units=[62,32],mlp_dropout=0.4,dropout=0.25,)

    model1.compile(loss="mse",optimizer=keras.optimizers.SGD(learning_rate=1e-4),metrics=[tf.keras.metrics.RootMeanSquaredError(),"acc"],)

    return model1

#MODEL BI_LSTM      
def getmodel():        
    model = layers.Sequential()
    forward_layer = layers.GRU(60, return_sequences=True,kernel_regularizer=tf.keras.regularizers.L1(0.02))
    backward_layer = layers.LSTM(34, return_sequences=True, go_backwards=True)
    model.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(window_size, maincolumn)))
    model.add(layers.Conv1D(20, kernel_size=2))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(16,return_sequences=True))
    model.add(layers.Conv1D(8, kernel_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(4))
    model.add(layers.Dense(o))



    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(),"acc"])
    model.build()
    return model

#rescaling
def getinversescaler(pred,Sc):
    forcaste=np.repeat(pred,maincolumn,axis=1)
   
    forcaste=forcaste[:,:maincolumn]
    
    pred=Sc.inverse_transform(forcaste)
   
    return pred



#opinion from web
def getrecommendation(ticker,timerinterval):
    
    tesla = TA_Handler(
        symbol=ticker,
        screener="forex",
        exchange="FX_IDC",
        interval=timerinterval,
    )
    return ta_select[tesla.get_analysis().summary["recommendation".upper()].upper()]



# def technicalIndicators(data):
#     data["MA"]=ta.MA(data['Close'], timeperiod=14, matype=0)
#     data['Close_wave'],data['High_wave'],data['Low_wave'],\
#     data['Open_wave']=wave_process.denoising([data['Close'],data['High'],data['Low'],data['Open']])
    
#     data['macd'], data['macdsignal'], data['macdhist'] = ta.MACD(data['Close'], \
#                                                                  fastperiod=7, slowperiod=14, signalperiod=5)
  
#     data['CCI14'] = ta.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
   

    
#     data['ATR14'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
#     data['NATR14']=ta.NATR(data['High'], data['Low'], data['Close'], timeperiod=14)
#     data["DEMA"]=ta.DEMA(data['Close'], timeperiod=14)

#     data["BBupper"],data["BBmidle"],data["BBlower"]=ta.BBANDS(data['Close'], timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
#     data["EMA10"]=ta.EMA(data['Close'], timeperiod=10)
#     data["EMA30"]=ta.EMA(data['Close'], timeperiod=30)
#     data["MOM"] = ta.MOM(data['Close'], timeperiod=14)
    
#     data['ROC']=ta.ROC(data['Close'], timeperiod=14)
#     data['KAMA']=ta.KAMA(data['Close'], timeperiod=14)

#     data["WillR"]=ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
    
#     # data["Pattern"]=ta.CDLCOUNTERATTACK(data['Open'],data['High'], data['Low'], data['Close'])



#     return data
