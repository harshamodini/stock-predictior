window_size=120
batch_size=120
ma_period=30
maincolumn=11
lop=1
v="index" if lop==1 else  "Datetime"
o=1
seed = 42
model_pathtransformer='models/{}-{}'.format("transformer","1h")
model_pathbi='models/{}-{}'.format("bi-lstm","1h")

ta_select={"STRONG_BUY":1,"BUY":1,"NEUTRAL":0,"SELL":-1,"STRONG_SELL":-1}
timeframe=["1m","5m","15m", "30m","1h", "2h", "4h", "1d","1W","1M"]