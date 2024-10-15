from dotenv import load_dotenv
load_dotenv(override=True)

import pandas as pd
import pandas_ta as ta
import os
import sys
from pathlib import Path
local_pth = Path(os.path.abspath(__file__)).parent
sys.path.append(str(local_pth))
sys.path.append(str(local_pth.parent))
sys.path.append(str(local_pth.parent.parent))
import pickle
from datetime import datetime, timedelta
import json
from collections import OrderedDict
from llmsignal_v2.pivots import PivotPoint
from fundamental.cpy_base import cpy_profiling 
# from priceLoader.finnhub_load import price_fetch
# from priceLoader.polygon_load import price_fetch
from priceLoader.finnhub_db_load import price_fetch

from analyst_rec.recommend import analyst_hist
from sentiment.senti import senti_hist_v2 as senti_hist
from chartist_pattern.chart_pattern import chart_pattern
from chartist_pattern.chartist_img import chartist_pattern_draw
import logging
import pytz

logging.basicConfig(format = '%(asctime)s - %(name)s: %(filename)s - Line:%(lineno)d, %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def rsi(args: dict):
    length = args.get("length", 14)
    
    # symbol = args["symbol"]

    signal_indicators = args.get("signal_indicators", True)
    close = args.get("close", None)
    assert close is not None, f"`close` price DataFrame missing" 

    #set xa, xb threshold, default xa=80, xb=20
    args["xa"] = args.get("xa", 70) 
    args["xb"] = args.get("xb", 30)

    tv = ta.rsi(**args)
    tv.dropna(inplace = True)

    ky1 = f"RSI_{args['length']}_B_20"
    ky2 = f"RSI_{args['length']}_A_80"
    #get signal
    def signal_gen(x):
        if eval(f"x.{ky1}") == 1:
            return "Buy"
        if eval(f"x.{ky2}") == 1:
            return "Sell"
        return "Hold"
    tv["action"] = tv.apply(signal_gen, axis = 1)
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    args.pop("close")
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": f"RSI_{length}",
        "value_full": json.dumps(val),
        "value": tv[f"RSI_{length}"],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** RSI done")
    return ret_out

def stochrsi(args):
    close = args.get("close", None)
    assert close is not None, f"`close` price DataFrame missing"  
    # symbol = args["symbol"]  
    tv = ta.stochrsi(**args)
    tv.dropna(inplace = True)

    ky1 = f"STOCHRSIk_{args['length']}_{args['rsi_length']}_{args['k']}_{args['d']}"
    ky2 = f"STOCHRSId_{args['length']}_{args['rsi_length']}_{args['k']}_{args['d']}"
    # sngv = tv[ky1] - tv[ky2] #k-cross-d
    tv["kd_diff"] = tv[ky1] - tv[ky2] #k-cross-d
    def signal_gen(x):
        # print(x)
        # if eval(f"x.{ky1}") > 80 and eval(f"x.{ky2}") > 80 and x.kd_diff < 0:
        if eval(f"x.{ky1}") > 80 and x.kd_diff < 0:
            return "Sell"
        # if eval(f"x.{ky1}") < 20 and eval(f"x.{ky2}") < 20 and x.kd_diff > 0:
        if eval(f"x.{ky1}") < 20 and x.kd_diff > 0:
            return "Buy"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": f"stochrsi_{args['length']}_{args['rsi_length']}_{args['k']}_{args['d']}",
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** stochrsi done")
    return ret_out

def stoch(args):
    close = args.get("close", None)
    high = args.get("high", None)
    low = args.get("low", None)
    assert close is not None, f"`close` price DataFrame missing"    
    assert high is not None, f"`high` price DataFrame missing"    
    assert low is not None, f"`low` price DataFrame missing"    
    # symbol = args["symbol"]

    tv = ta.stoch(**args)
    tv.dropna(inplace = True)

    ky1 = f"STOCHk_{args['k']}_{args['d']}_{args['smooth_k']}"
    ky2 = f"STOCHd_{args['k']}_{args['d']}_{args['smooth_k']}"
    # sngv = tv[ky1] - tv[ky2] #k-cross-d
    tv["kd_diff"] = tv[ky1] - tv[ky2] #k-cross-d
    def signal_gen(x):
        # print(x)
        # if eval(f"x.{ky1}") > 80 and eval(f"x.{ky2}") > 80 and x.kd_diff < 0:
        if eval(f"x.{ky1}") > 80 and x.kd_diff < 0:
            return "Sell"
        # if eval(f"x.{ky1}") < 20 and eval(f"x.{ky2}") < 20 and x.kd_diff > 0:
        if eval(f"x.{ky1}") < 20 and x.kd_diff > 0:
            return "Buy"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    del args['high']
    del args['low']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": f"stoch_{args['k']}_{args['d']}_{args['smooth_k']}",
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** stoch done")

    return ret_out

def cci(args):
    close = args.get("close", None)
    high = args.get("high", None)
    low = args.get("low", None)
    assert close is not None, f"`close` price DataFrame missing"    
    assert high is not None, f"`high` price DataFrame missing"    
    assert low is not None, f"`low` price DataFrame missing"    

    # symbol = args["symbol"]
    tv = ta.cci(**args).to_frame()
    tv.dropna(inplace = True)

    ky1 = f"CCI_{args['length']}_0.015"
    tv["yesterday"] = tv[ky1].shift(1)
    def signal_gen(x):
        # print(x)
        if eval(f"x['{ky1}']") > 100:#  and x.yesterday > 100:
            return "Sell"
        if eval(f"x['{ky1}']") < -100: # and x.yesterday < -100:
            return "Buy"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    del args['high']
    del args['low']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": f"cci_{args['length']}",
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** cci done")

    return ret_out

def adx(args):
    close = args.get("close", None)
    high = args.get("high", None)
    low = args.get("low", None)
    assert close is not None, f"`close` price DataFrame missing"    
    assert high is not None, f"`high` price DataFrame missing"    
    assert low is not None, f"`low` price DataFrame missing"    

    # symbol = args["symbol"]
    tv = ta.adx(**args)
    tv.dropna(inplace = True)

    ky1 = f"ADX_{args['lensig']}"
    ky2 = f"DMP_{args['length']}"
    ky3 = f"DMN_{args['length']}"
    tv["yesterday"] = tv[ky1].shift(1)
    tv["dmp_diff"] = tv[ky2] - tv[ky3]
    def signal_gen(x):
        # print(x)
        if eval(f"x['{ky1}']") > 25 and x.dmp_diff > 0:
            return "Buy"
        if eval(f"x['{ky1}']") > 25 and x.dmp_diff < 0:
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    del args['high']
    del args['low']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": f"adx_{args['length']}",
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** adx done")

    return ret_out

def ao(args):
    high = args.get("high", None)
    low = args.get("low", None)
    assert high is not None, f"`high` price DataFrame missing"    
    assert low is not None, f"`low` price DataFrame missing"    
    # symbol = args["symbol"]

    tv = ta.ao(**args).to_frame()
    tv.dropna(inplace = True)

    ky1 = f"AO_{args['fast']}_{args['slow']}"
    tv["yesterday"] = tv[ky1].shift(1)
    def signal_gen(x):
        # print(x)
        if eval(f"x['{ky1}']") > 0 and x.yesterday < 0:
            return "Buy"
        if eval(f"x['{ky1}']") < 0 and x.yesterday > 0:
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['high']
    del args['low']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** ao done")

    return ret_out

def mom(args):
    close = args.get("close", None)
    assert close is not None, f"`close` price DataFrame missing"    
    # symbol = args["symbol"]

    tv = ta.mom(**args).to_frame()
    tv.dropna(inplace = True)

    ky1 = f"MOM_{args['length']}"
    tv["yesterday"] = tv[ky1].shift(1)
    def signal_gen(x):
        # print(x)
        if eval(f"x['{ky1}']") > 0:# and x.yesterday < 0:
            return "Buy"
        if eval(f"x['{ky1}']") < 0: # and x.yesterday > 0:
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** MOM done")

    return ret_out


def macd(args):
    close = args.get("close", None)
    assert close is not None, f"`close` price DataFrame missing"    
    # symbol = args["symbol"]

    tv = ta.macd(**args)
    tv.dropna(inplace = True)

    ky1 = f"MACDh_{args['fast']}_{args['slow']}_{args['signal']}_XA_0" #cross-over buy
    ky2 = f"MACDh_{args['fast']}_{args['slow']}_{args['signal']}_XB_0" #cross-over sell
    ky3 = f"MACD_{args['fast']}_{args['slow']}_{args['signal']}" #MACD line
    ky4 = f"MACDh_{args['fast']}_{args['slow']}_{args['signal']}" #MACD line - signal
    def signal_gen(x):
        # print(x)
        if eval(f"x['{ky3}']") > 0 and eval(f"x['{ky4}']") > 0:
            return "Buy"
        if eval(f"x['{ky3}']") > 0 and eval(f"x['{ky4}']") < 0:
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv[ky3],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** MACD done")

    return ret_out

def willr(args):
    close = args.get("close", None)
    high = args.get("high", None)
    low = args.get("low", None)
    assert close is not None, f"`close` price DataFrame missing" 
    assert high is not None, f"`high` price DataFrame missing"  
    assert low is not None, f"`low` price DataFrame missing"     
    # symbol = args["symbol"]

    tv = ta.willr(**args).to_frame()
    tv.dropna(inplace = True)

    ky1 = f"WILLR_{args['length']}"
    def signal_gen(x):
        # print(x)
        if eval(f"x['{ky1}']") < -80:
            return "Buy"
        if eval(f"x['{ky1}']") > -20: 
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    del args['high']
    del args['low']

    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** WILLR done")

    return ret_out

def eri(args):
    close = args.get("close", None)
    high = args.get("high", None)
    low = args.get("low", None)
    assert close is not None, f"`close` price DataFrame missing" 
    assert high is not None, f"`high` price DataFrame missing"  
    assert low is not None, f"`low` price DataFrame missing"     

    # symbol = args["symbol"]

    tv = ta.eri(**args)#.to_frame()
    tv.dropna(inplace = True)

    ky1 = f"BULLP_{args['length']}"
    ky2 = f"BEARP_{args['length']}"
    ky3 = f"BPBE_SUM_{args['length']}"
    # tv[f"{ky1}_MAX"] = tv[ky1].shift(1).rolling(args['length']).max()
    # tv[f"{ky2}_MIN"] = tv[ky2].shift(1).rolling(args['length']).min()
    tv[ky3] = tv[ky1] + tv[ky2]
    tv[f"{ky3}_MAX"] = tv[ky3].shift(1).rolling(args['length']).max()
    tv[f"{ky3}_MIN"] = tv[ky3].shift(1).rolling(args['length']).min()

    def signal_gen(x):
        # print(x)
        # if eval(f"x['{ky2}']") < 0 and eval(f"x['{ky2}']") > eval(f"x['{ky2}_MIN']"):
        if eval(f"x['{ky3}']") < 0 and eval(f"x['{ky3}']") > eval(f"x['{ky3}_MIN']"):
            return "Buy"
        # if eval(f"x['{ky1}']") > 0 and eval(f"x['{ky1}']") > eval(f"x['{ky1}_MAX']"):
        if eval(f"x['{ky3}']") > 0 and eval(f"x['{ky3}']") > eval(f"x['{ky3}_MAX']"):
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    del args['high']
    del args['low']

    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv[ky3],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** ERI done")

    return ret_out

def uo(args):
    close = args.get("close", None)
    high = args.get("high", None)
    low = args.get("low", None)
    assert close is not None, f"`close` price DataFrame missing" 
    assert high is not None, f"`high` price DataFrame missing"  
    assert low is not None, f"`low` price DataFrame missing"     

    # symbol = args["symbol"]

    tv = ta.uo(**args).to_frame()
    tv.dropna(inplace = True)

    ky1 = f"UO_{args['fast']}_{args['medium']}_{args['slow']}"
    ky2 = f"UO_{args['fast']}_{args['medium']}_{args['slow']}_MIN"
    ky3 = f"UO_{args['fast']}_{args['medium']}_{args['slow']}_MAX"
    ky4 = "h_max"
    ky5 = "l_min"
    tv['high'] = high
    tv['low'] = low
    tv[ky2] = tv[ky1].shift(1).rolling(args['medium']).min()
    tv[ky3] = tv[ky1].shift(1).rolling(args['medium']).max()
    tv[ky4] = tv["high"].shift(1).rolling(args['medium']).max()
    tv[ky5] = tv["low"].shift(1).rolling(args['medium']).min()
    def signal_gen(x):
        if eval(f"x['{ky1}']") > eval(f"x['{ky2}']") and x.low < eval(f"x['{ky5}']"):
            return "Buy"
        if eval(f"x['{ky1}']") < eval(f"x['{ky3}']") and x.high > eval(f"x['{ky4}']"):
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    del args['high']
    del args['low']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** UO done")

    return ret_out

def sma(args):
    close = args.get("close", None)
    assert close is not None, f"`close` price DataFrame missing" 

    # symbol = args["symbol"]

    tv = ta.sma(**args).to_frame()
    tv.dropna(inplace = True)

    ky1 = f"SMA_{args['length']}"
    ky2 = f"SMA_{args['length']}_MIN"
    ky3 = f"SMA_{args['length']}_MAX"
    tv['close'] = close
    tv[ky2] = tv[ky1].shift(1).rolling(args['length_sig']).min()
    tv[ky3] = tv[ky1].shift(1).rolling(args['length_sig']).max()

    def signal_gen(x):
        if x.close > eval(f"x['{ky1}']") and eval(f"x['{ky1}']") > eval(f"x['{ky2}']"):
            return "Buy"
        if x.close < eval(f"x['{ky1}']") and eval(f"x['{ky1}']") < eval(f"x['{ky3}']"):
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** SMA-{args['length']} done")

    return ret_out

def ema(args):
    close = args.get("close", None)
    assert close is not None, f"`close` price DataFrame missing" 

    # symbol = args["symbol"]

    tv = ta.ema(**args).to_frame()
    tv.dropna(inplace = True)

    ky1 = f"EMA_{args['length']}"
    ky2 = f"EMA_{args['length']}_MIN"
    ky3 = f"EMA_{args['length']}_MAX"
    tv['close'] = close
    tv[ky2] = tv[ky1].shift(1).rolling(args['length_sig']).min()
    tv[ky3] = tv[ky1].shift(1).rolling(args['length_sig']).max()

    def signal_gen(x):
        if x.close > eval(f"x['{ky1}']") and eval(f"x['{ky1}']") > eval(f"x['{ky2}']"):
            return "Buy"
        if x.close < eval(f"x['{ky1}']") and eval(f"x['{ky1}']") < eval(f"x['{ky3}']"):
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }
    logger.info(f"** EMA-{args['length']} done")

    return ret_out

def vwma(args):
    close = args.get("close", None)
    assert close is not None, f"`close` price DataFrame missing" 

    # symbol = args["symbol"]

    tv = ta.vwma(**args).to_frame()
    tv.dropna(inplace = True)

    ky1 = f"VWMA_{args['length']}"
    ky2 = f"VWMA_{args['length']}_MIN"
    ky3 = f"VWMA_{args['length']}_MAX"
    tv['close'] = close
    tv[ky2] = tv[ky1].shift(1).rolling(args['length_sig']).min()
    tv[ky3] = tv[ky1].shift(1).rolling(args['length_sig']).max()

    def signal_gen(x):
        if x.close > eval(f"x['{ky1}']") and eval(f"x['{ky1}']") > eval(f"x['{ky2}']"):
            return "Buy"
        if x.close < eval(f"x['{ky1}']") and eval(f"x['{ky1}']") < eval(f"x['{ky3}']"):
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }

    logger.info(f"** VWMA done")

    return ret_out

def hma(args):
    close = args.get("close", None)
    assert close is not None, f"`close` price DataFrame missing" 

    # symbol = args["symbol"]

    tv = ta.hma(**args).to_frame()
    tv.dropna(inplace = True)

    ky1 = f"HMA_{args['length']}"
    ky2 = f"HMA_{args['length']}_MIN"
    ky3 = f"HMA_{args['length']}_MAX"
    tv['close'] = close
    tv[ky2] = tv[ky1].shift(1).rolling(args['length_sig']).min()
    tv[ky3] = tv[ky1].shift(1).rolling(args['length_sig']).max()

    def signal_gen(x):
        if x.close > eval(f"x['{ky1}']") and eval(f"x['{ky1}']") > eval(f"x['{ky2}']"):
            return "Buy"
        if x.close < eval(f"x['{ky1}']") and eval(f"x['{ky1}']") < eval(f"x['{ky3}']"):
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv[ky1],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }

    logger.info(f"** HMA done")

    return ret_out

def ichimoku(args):
    close = args.get("close", None)
    high = args.get("high", None)
    low = args.get("low", None)
    assert close is not None, f"`close` price DataFrame missing" 
    assert high is not None, f"`high` price DataFrame missing"  
    assert low is not None, f"`low` price DataFrame missing"     
    #tenkan:9,kijun:26,senkou:52
    #(h+l)/2: average on 9,26,52
    # symbol = args["symbol"]

    tv, spandf = ta.ichimoku(**args)
    tv.dropna(inplace = True)

    tv['close'] = close
    ky1 = f"ISA_{args['tenkan']}" #9+26->shift-26
    ky2 = f"ISB_{args['kijun']}" #52 ->shift-26
    ky3 = f"ITS_{args['tenkan']}" #9
    ky4 = f"IKS_{args['kijun']}" #26
    def signal_gen(x):
        if (
            eval(f"x['{ky1}']") > eval(f"x['{ky2}']")  #uptrebd
            and x.close > eval(f"x['{ky1}']") #crossover
        ):
            return "Buy"
        if (
            eval(f"x['{ky1}']") < eval(f"x['{ky2}']")  #downtrend
            and x.close < eval(f"x['{ky1}']") #crossover
        ):
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    del args['high']
    del args['low']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv[ky4],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }

    logger.info(f"** ichimoku done")

    return ret_out

def bbands(args):
    close = args.get("close", None)
    assert close is not None, f"`close` price DataFrame missing" 

    # symbol = args["symbol"]

    #tenkan:9,kijun:26,senkou:52
    #(h+l)/2: average on 9,26,52
    tv = ta.bbands(**args) #["close"], length = 20, std = 2.0)
    tv.dropna(inplace = True)

    tv['close'] = close
    ky1 = f"BBU_{args['length']}_{args['std']}" #upper band
    ky2 = f"BBL_{args['length']}_{args['std']}" #lower band
    ky3 = f"BBM_{args['length']}_{args['std']}" #middle band

    tv["std"] = (tv[ky1] - tv[ky2]) / (args["std"] * 2.0)
    tv["zscore"] = (tv["close"] - tv[ky3]) / tv["std"]
    # tv["zscore_ma"] = tv["zscore"].shift(1).rolling(args["length_sig"]).mean()
    tv["zscore_ma"] = ta.ema(tv["zscore"].shift(1), args["length_sig"])
    def signal_gen(x):
        if (
            (x.zscore > x.zscore_ma
            and x.zscore_ma < 0.0
            and x.zscore > 0.0) #crossover
            or (
                x.zscore_ma < -1.0
                and x.zscore > -1.0
            )
        ):
            return "Buy"

        if (
            (
                x.zscore < x.zscore_ma
                and x.zscore_ma >= 1.0
                and x.zscore <= 1.0) #crossover
                or (
                    x.zscore > 2.0
                )
        ):
            return "Sell"
        return "Hold"

    tv["action"] = tv.apply(signal_gen, axis = 1)
    #
    out = tv.tail(1).to_dict(orient = "index")
    val = list(out.values())[0]
    del args['close']
    ret_out = {
        "date": list(out.keys())[0],
        # "symbol": symbol,
        "ta_name": ky1.lower(),
        "value_full": json.dumps(val),
        "value": tv["zscore"],
        "action": tv["action"],
        "param": args,
        "full": tv,
    }

    logger.info(f"** BBANDS done")

    return ret_out

def price2techIndicator(symbol: str, price_df: pd.DataFrame, ta_cfg: dict):
    ti_values = {}
    for cat, cat_cfg in ta_cfg.items():
        ti_values_cat = []
        for ta_name, cfg in cat_cfg.items():
            args = cfg["args"]
            desc = cfg["desc"]
            # assert type(args) == type(desc), f"*** Warning: args, desc MUST be same type, both dict or list"
            if not isinstance(args, list):
                args = [args]
            if not isinstance(desc, list):
                desc = [desc]
            for arg_now, desc_now in zip(args, desc):
                for col in arg_now["price"]:
                    arg_now[col] = price_df[col]

                tmp_value_df = pd.Series(data = [None] * arg_now[arg_now["price"][0]].shape[0], index = arg_now[arg_now["price"][0]].index)

                arg_now.pop("price") 
                # tv = globals()[ta_name](arg_now)
                if cat == "pivots":
                    cols = ["s3", "s2", "s1", "p", "r1", "r2", "r3"]
                    try:
                        tv = globals()[ta_name](arg_now)
                        out_fmt = tv[cols]
                        out_fmt["symbol"] = symbol
                        out_fmt["desc"] = desc_now
                    except:
                        tmp_act_df = pd.DataFrame(index = arg_now["close"].index, columns=cols)
                        for v in cols:
                            tmp_act_df[v] = None
                        out_fmt = tmp_act_df
                        out_fmt["symbol"] = symbol
                        out_fmt["desc"] = desc_now
                else:
                    try:
                        tv = globals()[ta_name](arg_now)
                        out_fmt = pd.DataFrame(
                            {
                                "symbol": symbol,
                                "value": tv["value"], 
                                "action": tv["action"],
                                "desc": desc_now,
                            }
                        )
                    except:
                        out_fmt = pd.DataFrame(
                            {
                                "symbol": symbol,
                                "value": tmp_value_df, 
                                "action": tmp_value_df,
                                "desc": desc_now,
                            }
                        )

                ti_values_cat.append(out_fmt)
                for ky in ["high", "close", "open", "low", "volume"]:
                    if arg_now.get(ky, None) is not None:
                        arg_now.pop(ky)
                logger.info(f"{desc_now} -> {arg_now}: finished")
        ti_values[cat] = pd.concat(ti_values_cat, axis = 0)
    ti_values["latest_dt"] = price_df.index[-1]
    ti_values["price_df"] = price_df

    return  ti_values


def price2techIndicator_v1(symbol: str, price_df: pd.DataFrame):
    """
    Oscillators TA
    """
    osc_stack = []
    #RSI
    win_len = 14
    args = {
        "close": price_df["close"], #.tail(win_len + 1),
        "length": win_len,
        "signal_indicators": True,
    }
    tv = rsi(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
         "action": tv["action"],
         "desc": f"Relative Strength Index ({args['length']})"
    }
    )
    osc_stack.append(out_fmt)

    #Stochastic RSI Fast (3, 3, 14, 14)
    args = {
        "close": price_df["close"],#.tail(14*2 + 3 + 1),
        "length": 14,
        "rsi_length": 14,
        "k": 3,
        "d": 3,
    }
    tv = stochrsi(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
         "action": tv["action"],
         "desc": f"Stochastic RSI Fast ({args['k']},{args['d']},{args['length']},{args['rsi_length']})"
    }
    )
    osc_stack.append(out_fmt)

    #Stochastic
    args = {
        "high": price_df["high"],
        "low": price_df["low"],
        "close": price_df["close"],
        "k": 14,
        "d": 3,
        "smooth_k": 3,
    }
    tv = stoch(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
         "action": tv["action"],
         "desc": f"Stochastic Oscillator ({args['k']},{args['d']},{args['smooth_k']})"
    }
    )
    osc_stack.append(out_fmt)

    #Commodity Channel Index (CCI) 
    args = {
        "high": price_df["high"],
        "low": price_df["low"],
        "close": price_df["close"],
        "length": 20,
    }
    tv = cci(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
         "action": tv["action"],
         "desc": f"Commodity Channel Index ({args['length']})"
    }
    )
    osc_stack.append(out_fmt)

    #Average Directional Index
    args = {
        "high": price_df["high"],
        "low": price_df["low"],
        "close": price_df["close"],
        "length": 14,
        "lensig": 14, 
    }
    tv = adx(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
        "action": tv["action"],
        "desc": f"Average Directional Index ({args['length']})"
    }
    )
    osc_stack.append(out_fmt)

    #Awesome Oscillator (AO)
    args = {
        "high": price_df["high"],
        "low": price_df["low"],
        "fast": 5,
        "slow": 34, 
    }
    tv = ao(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
        "action": tv["action"],
        "desc": f"Awesome Oscillator ({args['fast']}, {args['slow']})"
    }
    )
    osc_stack.append(out_fmt)

    # Momentum Indicator
    args = {
        "close": price_df["close"],
        "length": 10,
    }
    tv = mom(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
        "action": tv["action"],
        "desc": f"Momentum Indicator ({args['length']})"
    }
    )
    osc_stack.append(out_fmt)

    # MACD
    args = {
        "close": price_df["close"],
        "fast": 12,
        "slow": 26,
        "signal": 9, 
        "signal_indicators": True,
    }
    tv = macd(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
        "action": tv["action"],
        "desc": f"MACD Level ({args['fast']},{args['slow']})"
    }
    )
    osc_stack.append(out_fmt)

    # Williams %R 
    args = {
        "high": price_df["high"],
        "low": price_df["low"],
        "close": price_df["close"],
        "length": 14,
    }
    tv = willr(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
        "action": tv["action"],
        "desc": f"Williams Percent Range ({args['length']})"
    }
    )
    osc_stack.append(out_fmt)

    # Bears Power oscillator (Elder Ray Index (ERI)
    args = {
        "high": price_df["high"],
        "low": price_df["low"],
        "close": price_df["close"],
        "length": 13,
    }
    tv = eri(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
        "action": tv["action"],
        "desc": f"Bull Bear Power ({args['length']})"
    }
    )
    osc_stack.append(out_fmt)

    # Ultimate Oscillator
    args = {
        "high": price_df["high"],
        "low": price_df["low"],
        "close": price_df["close"],
        "fast": 7,
        "medium": 14,
        "slow": 28,
    }
    tv = uo(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
        "action": tv["action"],
        "desc": f"Ultimate Oscillator ({args['fast']}, {args['medium']}, {args['slow']})"
    }
    )
    osc_stack.append(out_fmt)

    #Moverage average
    ma_stack = []
    # SMA
    win_lens = [10, 20, 30, 50, 100, 200]
    for length in win_lens:
        args = {
            "close": price_df["close"],
            "length": length,
            "length_sig": 14, #lookback window to get signal
        }
        tv = sma(args)
        out_fmt = pd.DataFrame(
        {
            "symbol": symbol,
            "value": tv["value"], 
            "action": tv["action"],
            "desc": f"Simple Moving Average ({args['length']})"
        }
        )
        ma_stack.append(out_fmt)

    # EMA
    for length in win_lens:
        args = {
            "close": price_df["close"],
            "length": length,
            "length_sig": 14,
        }
        tv = ema(args)
        out_fmt = pd.DataFrame(
        {
            "symbol": symbol,
            "value": tv["value"], 
            "action": tv["action"],
            "desc": f"Exponential Moving Average ({args['length']})"
        }
        )
        ma_stack.append(out_fmt)

    # volume Weighted Moving Average (VWMA)
    args = {
        "close": price_df["close"],
        "volume": price_df["volume"],
        "length": 20,
        "length_sig": 14,
    }
    tv = vwma(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
        "action": tv["action"],
        "desc": f"Volume Weighted Moving Average ({args['length']})"
    }
    )
    ma_stack.append(out_fmt)

    # Hull Moving Average (HMA)
    args = {
        "close": price_df["close"],
        "length": 9,
        "length_sig": 14,
    }
    tv = hma(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
        "action": tv["action"],
        "desc": f"Hull Moving Average ({args['length']})"
    }
    )
    ma_stack.append(out_fmt)

    # Ichimoku Base Line (9, 26, 52, 26) Ichimoku Kinkō Hyō (Ichimoku)
    args = {
        "close": price_df["close"],
        "high": price_df["high"],
        "low": price_df["low"],
        "tenkan": 9,
        "kijun": 26,
        "senkou": 52,
        "include_chikou": False
    }
    tv = ichimoku(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
        "action": tv["action"],
        "desc": f"Ichimoku Cloud ({args['tenkan']}, {args['kijun']}, {args['senkou']}, {args['kijun']})"
    }
    )
    ma_stack.append(out_fmt)

    #Bollinger Bands (BBANDS)
    args = {
        "close": price_df["close"],
        "length": 20,
        "length_sig": 5,
        "std": 2.0
    }
    tv = bbands(args)
    out_fmt = pd.DataFrame(
    {
        "symbol": symbol,
        "value": tv["value"], 
        "action": tv["action"],
        "desc": f"Bollinger Bands ({args['length']}, {args['std']})"
    }
    )
    osc_stack.append(out_fmt)

    #pivots
    # price_df.index = price_df.index.map(lambda x: pytz.timezone("UTC").localize(x))
    pivot_pp = []
    for method in ["classic","fibonacci", "woodie", "demark", "camarilla"]:
        tv = PivotPoint(price_df, method = method)
        cols = ["s3", "s2", "s1", "p", "r1", "r2", "r3"]
        out_fmt = tv[cols]
        out_fmt["symbol"] = symbol
        out_fmt["desc"] = f"{method.title()}"
        logger.info(f"** Pivots - {method} done")

        pivot_pp.append(out_fmt)

    osc_stack = pd.concat(osc_stack, axis = 0)
    ma_stack = pd.concat(ma_stack, axis = 0)
    pivot_pp = pd.concat(pivot_pp, axis = 0)

    return {
        "oscillators": osc_stack,
        "moving_averages": ma_stack,
        "pivots": pivot_pp,
        "latest_dt": price_df.index[-1],
        "price_df": price_df,
    }

def prompt_df(ta_dfs: dict):
    #prepare TA prompt
    prompt_text = "## Technical Analys Indicators\n\n"
    price_df = ta_dfs["price_df"]
    osc_stack = ta_dfs.get("oscillators", None)
    if osc_stack is not None: 
        ta_prompt_osc = osc_stack.loc[price_df.index[-1]].reset_index()[["desc","value","action"]].set_index("desc")
        ta_prompt_osc = ta_prompt_osc.round(2)
        prompt_text += f"### Oscillators: the following table shows oscillator method, value, and signal recommendation\n\n"
        prompt_text += f"{ta_prompt_osc}\n\n"
    ma_stack = ta_dfs.get("moving_averages", None)
    if ma_stack is not None: 
        ta_prompt_ma = ma_stack.loc[price_df.index[-1]].reset_index()[["desc","value","action"]].set_index("desc")
        ta_prompt_ma = ta_prompt_ma.round(2)
        prompt_text += f"### Moving Averages: the following table shows moving average method, value, and signal recommendation\n\n"
        prompt_text += f"{ta_prompt_ma}\n\n"
        
    pivot_pp = ta_dfs.get("pivots", None)  
    if pivot_pp is not None:    
        ta_prompt_pivot = pivot_pp.loc[price_df.index[-1]].reset_index()[["desc","s3","s2","s1","p","r1","r2","r3"]].set_index("desc").transpose()
        ta_prompt_pivot = ta_prompt_pivot.round(2)
        prompt_text += f"### Pivots: the following table shows pivot point calculation method (Column), and support (3 levels: s1, s2, s3), resistance level and value (3 level: r1, r2, r3), and pivot point value (p). NaN means the value is not available\n\n"
        prompt_text += f"{ta_prompt_pivot}\n\n"

    return prompt_text

def extra_resource_prompt(symbol, price_df, use_fundamental = True, use_news = True, use_analyst = True):
    cpy_foudment = cpy_profiling(symbol) if use_fundamental else None
    if cpy_foudment is not None:
        cpy_foudment_tdy = cpy_foudment.iloc[-1].dropna().to_dict()
        cpy_prompt = f"## Company Fundamental Metrics and Values (quarterly reported at {cpy_foudment.index[-1].strftime('%Y-%m_%d')}) are listed in the following with format: financial metric following financial value\n\n"
        # cpy_prompt += "Metric\tValue\n"
        cpy_prompt += "\n".join(map(lambda x: f"{x[0]}\t{x[1]}", cpy_foudment_tdy.items()))
        use_fundamental = True
    else:
        # cpy_prompt = None
        cpy_prompt = f"""## Company Fundamental Metrics and Values: \nNOT AVAILABLE\n\n"""
        use_fundamental = False

    dt_from = price_df.index[0].strftime("%Y-%m-%d")
    dt_to = price_df.index[-1].strftime("%Y-%m-%d")
    analyst_df = analyst_hist(symbol) if use_analyst else None
    if analyst_df is not None: 
        analyst_df = analyst_df[analyst_df.index <= dt_to]
        analyst_tdy = analyst_df.iloc[-1].dropna()[["strongBuy", "buy", "hold", "sell", "strongSell", "total"]].to_dict()
        analyst_prompt = f"## Analyst Recommendation (reported at {analyst_df.index[-1]})\n"
        analyst_prompt += "The following data shows how many analysts cover the stock in total and their recommendation (i.e. strongBuy, buy, hold, sell, and strongSell) distribution\n\n"
        # cpy_prompt += "Metric\tValue\n"
        analyst_prompt += "\n".join(map(lambda x: f"{x[0]}\t{x[1]}", analyst_tdy.items()))

        use_analyst = True
    else:
        # analyst_prompt = None
        analyst_prompt = f"""## Analyst Recommendation: \nNOT AVAILABLE\n\n"""
        use_analyst = False

    senti_df = senti_hist(symbol=symbol, dt_from = dt_from, dt_to = dt_to) if use_news else None
    cols = {"positive_freq": "positive", 
            "neutral_freq": "neutral", 
            "negative_freq": "negative",
            "total": "total",
    }
    if senti_df is not None:
        senti_tdy = senti_df[["positive_freq", "neutral_freq", "negative_freq", "total"]].rename(columns = cols).iloc[-1]

        senti_prompt = f"""## News Sentiment:\nThe data how many news covering the stock in total and sentiment (i.e. positive, neutral, and negative) distribution\n\n"""
        senti_prompt += f"""Positive\t{senti_tdy.positive}\nNeutral\t\t{senti_tdy.neutral}\nNegative\t{senti_tdy.negative}\n"""
        use_news = True
    else:
        # senti_prompt = None
        senti_prompt = f"""## News Sentiment:\n NOT AVAILABLE\n\n"""
        use_news = False

    return {
        "fundamental": cpy_prompt, 
        "analyst": analyst_prompt, 
        "news": senti_prompt,
        "status": (use_fundamental, use_news, use_analyst)
    }

#
def prepare_prompt(symbol, dt_to, window_len = 365, use_fundamental = True, use_news = True, use_analyst = True, ta_config = None, use_chartist_text = True, use_chartist_image = True, source_price = "finnhub"):
    assert ta_config is not None, f"ta_config is not set"

    dt_to = datetime.fromisoformat(dt_to)
    dt_from = dt_to - timedelta(days = window_len)
    if source_price == "finnhub":
        timeframe = "D" 
        df = price_fetch(symbol, timeframe, dt_from, dt_to)
        
        col_low = dict(map(lambda x: (x, x.lower()), df.columns))
        df.rename(columns = col_low, inplace = True)
    else:
        df = pd.DataFrame()
        df = df.ta.ticker(symbol)
        col_map = dict(map(lambda x: (x, x.lower()), list(df.columns)))
        df.rename(columns = col_map, inplace = True)
        msk = (df.index >= pytz.timezone("US/Eastern").localize(dt_from)) & (df.index <= pytz.timezone("US/Eastern").localize(dt_to))
        df = df[msk][["open","high","low","close","volume"]]
        # print(df.tail(3))

    #check last date in price df is equal to dt_to
    if df.index[-1].strftime("%Y-%m-%d") != dt_to.strftime("%Y-%m-%d"):
        logger.info(f"OHLCV data latest date {df.index[-1]} NOT MATCH query date {dt_to}")
        prompt_data = {
            "user_prompt": None,
            "price_df": None,
            "use_fundamental": None, 
            "use_news": None, 
            "use_analyst": None,
            "image": None
        }

        return prompt_data #None, None, (None, None, None)

    v = price2techIndicator(symbol, df, ta_config)

    user_prompt = ""
    prompt = prompt_df(v)
    user_prompt += prompt

    # Fetech chart pattern
    try:
        if use_chartist_text or use_chartist_image:
            chart_desc, drawing_data = chart_pattern(symbol, dt_to.strftime("%Y-%m-%d"))
        else:
            chart_desc, drawing_data = None, None
    except:
        chart_desc, drawing_data = None, None

    # if chart_desc is None:
    #     chart_desc = "NOT AVAILABLE"
    image_chartist = None
    if chart_desc:
        prompt_pattern = ""
        if use_chartist_text:
            prompt_pattern += f"""\n### Technical chart pattern
            {chart_desc}
            """
        if use_chartist_image:
            if use_chartist_text:
                prompt_pattern += "\nThe image of chart pattern is given.\n"
            else:
                prompt_pattern = f"""\n### Technical chart pattern
                The image of chart pattern is given.
                """

        user_prompt += prompt_pattern
        
        if drawing_data and use_chartist_image:
            try:
                image_chartist = chartist_pattern_draw(symbol, drawing_data, dt_to.strftime("%Y-%m-%d"), timeframe = "hour", interval = 1, save_image = None, enable_viewer = False, use_chartist_forecast = False)
                logger.info(f"** chartist image available -> {symbol}")
            except Exception as e:
                logger.info(f"** chartist image NOT AVAILABLE -> {symbol} -> {e}")    
    #

    extra_prompt = extra_resource_prompt(symbol, df, use_fundamental = use_fundamental, use_news = use_news, use_analyst = use_analyst) 
    for ky in ["fundamental", "news", "analyst"]:
        if extra_prompt[ky]:
            user_prompt += "\n\n" + extra_prompt[ky]
    
    prompt_data = {
        "user_prompt": user_prompt,
        "price_df": df,
        "use_fundamental": extra_prompt["status"][0], 
        "use_news": extra_prompt["status"][1], 
        "use_analyst": extra_prompt["status"][2],
        "image": image_chartist
    }
    # return user_prompt, df, extra_prompt["status"], image_chartist

    return prompt_data

if __name__ == "__main__":
    import toml
    symbol = "AAPL"
    dt_to = "2024-10-14"
    ta_config_file = str(Path(os.path.abspath(__file__)).parent/"config_ta.toml")
    ta_cfg = toml.load(ta_config_file)

    prompt_now = prepare_prompt(symbol, dt_to, window_len = 365, ta_config=ta_cfg, source_price="finnhub")
    print(prompt_now)

if __name__ == "__main1__":
    import toml
    ta_config = toml.load("/Users/shenggao/work/PilotAI-stockSignal/src/opt_strategy/config_ta.toml") 
    timeframe = "D" 
    symbol = "QQQ"
    dt_to = "2024-08-28"
    dt_to = datetime.fromisoformat(dt_to)
    dt_from = dt_to - timedelta(days = 365)
    df = price_fetch(symbol, timeframe, dt_from, dt_to)
    
    col_low = dict(map(lambda x: (x, x.lower()), df.columns))
    df.rename(columns = col_low, inplace = True)
    price2techIndicator(symbol, df, ta_config)

