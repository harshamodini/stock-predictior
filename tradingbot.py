

import pandas as pd
import numpy as np
from flask import Flask, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import joblib
from support_func  import *
import com_var
import logging
import random
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

prevpredmax=0

prevpredmin=1
models = {}
scalers = {}
testingcontext={"Symbolname":"gbpusd","Cost":0.00015000000000000001,"Pridiction_size":4,"Open":[1.32006,1.32011,1.31988,1.3205,1.31991,1.31913,1.31919,1.31925,1.31905,1.31838,1.31901,1.31818,1.31605,1.31749,1.31955,1.32023,1.31982,1.31799,1.31918,1.31838,1.31791,1.31711,1.31822,1.31905,1.31823,1.3183,1.31911,1.3194,1.31886,1.32127,1.32067,1.3206,1.32215,1.32183,1.32124,1.31896,1.31624,1.31642,1.31886,1.31976,1.31957,1.31867,1.32085,1.31908,1.31897,1.31947,1.31835,1.31896,1.31785,1.31798],"High":[1.32053,1.32095,1.32062,1.32126,1.32004,1.31928,1.31956,1.3195,1.31911,1.32014,1.32054,1.31925,1.31785,1.31964,1.32113,1.3214,1.31989,1.31941,1.32053,1.31896,1.31837,1.31834,1.31954,1.31911,1.31856,1.31919,1.31968,1.31984,1.32127,1.32133,1.32129,1.32248,1.32234,1.32197,1.32127,1.31925,1.31769,1.3189,1.32,1.3205,1.32001,1.32225,1.32087,1.31989,1.31948,1.31981,1.3193,1.31901,1.3182,1.31798],"Low":[1.32006,1.31954,1.31987,1.31969,1.31834,1.31872,1.31881,1.31834,1.31818,1.31838,1.31807,1.31575,1.3157,1.31711,1.31802,1.31867,1.31674,1.31668,1.31799,1.31776,1.31698,1.31671,1.31781,1.3183,1.31764,1.3183,1.31887,1.31834,1.31862,1.31996,1.32046,1.32059,1.32126,1.32061,1.31865,1.31605,1.31591,1.3163,1.31803,1.31926,1.31787,1.31816,1.31882,1.31807,1.31811,1.31829,1.31789,1.31775,1.31727,1.31798],"Close":[1.32014,1.31984,1.32048,1.31991,1.31912,1.31918,1.31929,1.31905,1.3184,1.31902,1.31821,1.31604,1.3175,1.31955,1.32023,1.31982,1.31801,1.31919,1.3184,1.3179,1.31711,1.31825,1.31906,1.31868,1.3183,1.31909,1.3194,1.31885,1.32127,1.32068,1.32059,1.32217,1.32183,1.32125,1.31896,1.31623,1.31642,1.31889,1.31975,1.31955,1.31865,1.32084,1.3191,1.31898,1.31944,1.31837,1.31898,1.31808,1.31783,1.31798]}


    
def get_model():
    key = model_pathtransformer
    if key not in models:
        models[key] = load_model(f'{key}')
    return models[key]
def getdataframe(context):
    data= pd.DataFrame({"Close":context["Close"]})
    data=technicalIndicators(data)
    data.dropna(how='any', inplace=True)
    data = data[data.shape[0] % batch_size:]
    data.reset_index(inplace=True)
    data.drop("index",axis=1,inplace=True)

    return data
    

def getpredictednew(next,data,model):
    newtable=data.copy(deep=True)
    for i in range(next):
        X=newtable.iloc[-window_size:,:].copy(deep=True)
        Scalerc=MinMaxScaler()
        X.iloc[:,:]=Scalerc.fit_transform(X)[:,:]
        Xn=np.asarray([X[:]])
        ypredn = model.predict(Xn)
        ypredn=getinversescaler(ypredn,Scalerc)[0:]
        newtable=newtable.append(pd.DataFrame(ypredn,index=newtable.iloc[-1:].index+1,columns=newtable.columns.to_list()))
    return newtable

app = Flask(__name__)
 
@app.route('/predict/', methods=['GET','POST'])
def predict():
    content = request.get_json(silent=True)


    model = get_model()
    df = getdataframe(content)
    df=getpredictednew(3,df,model)
    pred=df[-3:-2].Close.values[0]
    pred1=df[-1:].Close.values[0]
    


    
    global prevpredmax
    prevpredmax=max(abs(pred-pred1),prevpredmax)
    global prevpredmin
    prevpredmin=min(abs(pred1-pred),prevpredmin)
    cost = content["pipval"]*0.8
    print(pred1,pred,abs(pred1-pred),cost)
    

    if pred1-pred>=cost:
        print(1)

        return "-1";
    elif pred-pred1>=cost:
        print(-1)
        return "1";

    return "0"

if __name__ == '__main__':
    app.run(debug=True)
