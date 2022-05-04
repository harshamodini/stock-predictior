import talib as ta

def technicalIndicators(data):
    # print(data)
    data['macd'], data['macdsignal'], data['macdhist'] = ta.MACD(data['Close'], \
                                                                 fastperiod=7, slowperiod=14, signalperiod=5)
  
    data['CCI14'] = ta.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
   

    
    # data['ATR14'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['NATR14']=ta.NATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data["DEMA"]=ta.DEMA(data['Close'], timeperiod=14)

    data["BBupper"],data["BBmidle"],data["BBlower"]=ta.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data["EMA10"]=ta.EMA(data['Close'], timeperiod=10)
    data["EMA30"]=ta.EMA(data['Close'], timeperiod=30)
    data["MOM"] = ta.MOM(data['Close'], timeperiod=14)
    data["MA"]=ta.MA(data['Close'], timeperiod=14, matype=0)
    data['ROc']=ta.ROC(data['Close'], timeperiod=28)
    data["WillR"]=ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data["Pattern"]=ta.CDLCOUNTERATTACK(data['High'], data['Low'], data['Close'])



    return data

def add(data):
    

    data['RSI5'] = ta.RSI(data['Close_wave'], timeperiod=5)
    data['RSI10'] = ta.RSI(data['Close_wave'], timeperiod=10)
    data['RSI20'] = ta.RSI(data['Close_wave'], timeperiod=20)

    data['macd'], data['macdsignal'], data['macdhist'] = ta.MACD(data['Close_wave'], \
                                                                 fastperiod=12, slowperiod=26, signalperiod=9)

    data['slowk'], data['slowd'] = ta.STOCH(data['High_wave'], data['Low_wave'], \
                                            data['Close_wave'], fastk_period=5, slowk_period=3, slowk_matype=0,
                                            slowd_period=3, slowd_matype=0)
    data['fastk'], data['fastd'] = ta.STOCHF(data['High_wave'], data['Low_wave'], \
                                             data['Close_wave'], fastk_period=5, fastd_period=3, fastd_matype=0)

    data['WR5'] = ta.WILLR(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=5)
    data['WR10'] = ta.WILLR(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=10)
    data['WR20'] = ta.WILLR(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=20)

    data['ROC5'] = ta.ROC(data['Close_wave'], timeperiod=5)
    data['ROC10'] = ta.ROC(data['Close_wave'], timeperiod=10)
    data['ROC20'] = ta.ROC(data['Close_wave'], timeperiod=20)


    data['CCI5'] = ta.CCI(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=5)
    data['CCI10'] = ta.CCI(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=10)
    data['CCI20'] = ta.CCI(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=20)

    data['ATR5'] = ta.ATR(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=5)
    data['ATR10'] = ta.ATR(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=10)
    data['ATR20'] = ta.ATR(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=20)

    data['NATR5'] = ta.NATR(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=5)
    data['NATR10'] = ta.NATR(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=10)
    data['NATR20'] = ta.NATR(data['High_wave'], data['Low_wave'], data['Close_wave'], timeperiod=20)

    data['TRANGE'] = ta.TRANGE(data['High_wave'], data['Low_wave'], data['Close_wave'])
    return data


def add1(data):

    data['RSI5'] = ta.RSI(data['Close'], timeperiod=5)
    data['RSI10'] = ta.RSI(data['Close'], timeperiod=10)
    data['RSI20'] = ta.RSI(data['Close'], timeperiod=20)

    data['macd'], data['macdsignal'], data['macdhist'] = ta.MACD(data['Close'], \
                                                                 fastperiod=12, slowperiod=26, signalperiod=9)

    data['slowk'], data['slowd'] = ta.STOCH(data['High'], data['Low'], \
                                            data['Close'], fastk_period=5, slowk_period=3, slowk_matype=0,
                                            slowd_period=3, slowd_matype=0)
    data['fastk'], data['fastd'] = ta.STOCHF(data['High'], data['Low'], \
                                             data['Close'], fastk_period=5, fastd_period=3, fastd_matype=0)

    data['WR5'] = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=5)
    data['WR10'] = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=10)
    data['WR20'] = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=20)

    data['ROC5'] = ta.ROC(data['Close'], timeperiod=5)
    data['ROC10'] = ta.ROC(data['Close'], timeperiod=10)
    data['ROC20'] = ta.ROC(data['Close'], timeperiod=20)


    data['CCI5'] = ta.CCI(data['High'], data['Low'], data['Close'], timeperiod=5)
    data['CCI10'] = ta.CCI(data['High'], data['Low'], data['Close'], timeperiod=10)
    data['CCI20'] = ta.CCI(data['High'], data['Low'], data['Close'], timeperiod=20)

    data['ATR5'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=5)
    data['ATR10'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=10)
    data['ATR20'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=20)

    data['NATR5'] = ta.NATR(data['High'], data['Low'], data['Close'], timeperiod=5)
    data['NATR10'] = ta.NATR(data['High'], data['Low'], data['Close'], timeperiod=10)
    data['NATR20'] = ta.NATR(data['High'], data['Low'], data['Close'], timeperiod=20)

    data['TRANGE'] = ta.TRANGE(data['High'], data['Low'], data['Close'])
    return data