#*-* coding: utf-8 *-*
#!/usr/bin/env python3
"""
python ml_e2e* --target TQQQ  --model xgb --func realtime
python ml_e2e* --target TQQQ  --model xgb --func train --price_margin 0.01 --capital 100
python ml_e2e* --target TQQQ --func strategy --fromdate 2021-08-01 --todate 2022-10-04
python ml_e2e* --target TQQQ  --model xgb --func regressor
python ml_e2e* --target TQQQ  --model xgb --func regressor_predict
"""
# model_method = 'xgb' #
# model_method = 'catboost' #
import numpy as np
import pandas as pd
import math
from aimodel.featureOpt import signalClassifier
from aimodel.tickerDownloader import tickerDownloader
from aimodel.learnerRegressor import learnerRegressor 

from aimodel.mlStrategyLongshort import strategyEvaluate as strategyEvaluateLongshort
import pickle
# import xgboost as xgb
from catboost import CatBoostClassifier, Pool, cv
import datetime
import os
import json
import argparse
from aimodel.melog import setLog

from aimodel.learner import learner

from pathlib import Path
root_pth = str(Path(__file__).parent.absolute())
# print(root_pth)


def configure(args): #target_symbol, model_method, i_price_margin = 0.01, prefix = ''):
    if isinstance(args, dict):
        target_symbol = args['target']
        model_method = args['model']
        i_price_margin = args['price_margin']
        i_cash = args['capital']
    else:
        target_symbol = args.target
        model_method = args.model
        i_price_margin = args.price_margin
        i_cash = args.capital
    prefix = ''
 
    param = {
        'target': target_symbol,
        'cache_file': f'{root_pth}/{prefix}{target_symbol}_stock_cache.pkl',
        'cache_feat_file': f'{root_pth}/{prefix}{target_symbol}_stock_cache_feature.pkl',
        'refresh_cache': True, #not use cache and download
        'n_lookback_feat': 3, #how many dys lookback to prepare feature vector
        'price_margin': i_price_margin, #0.01,
        'label_price_source': 'Close',
        'lookback_window': (5, 10, 14, 20), #extract statistics
        'price_scaler': '', #'log', #'sqrt', #sqrt, ''
        # 'split_ratio': {'train': 0.7, 'dev': 0.2, 'eval': 0.1}, #learning model: all >0, and prediction: eval = 1.0, other = 0
        'split_ratio': {'train': 0.0, 'dev': 0.0, 'eval': 1.0}, #learning model: all >0, and prediction: eval = 1.0, other = 0
        'is_realtime': True,


        'model_save': f'{root_pth}/datas/{prefix}{target_symbol}_{model_method}_model.json',
        'model_acc_report': f'{root_pth}/datas/{prefix}{target_symbol}_{model_method}_model_accuracy_report.json',
        'stake': 1, #backtrader stake size to buy 
        'cash': i_cash,
        'is_onlylong': True, #true = long only
        'o_signal_file': f'{root_pth}/datas/{prefix}{target_symbol}_{model_method}_predict_eval.csv', #predicted signal file saved
        'o_log_file':f'{root_pth}/datas/{prefix}{target_symbol}_{model_method}_predict_eval_strategy.log',
        'o_quant_html': f'{root_pth}/datas/{prefix}{target_symbol}_{model_method}_predict_eval_strategy.html',
        'o_quant_file': f'{root_pth}/datas/{prefix}{target_symbol}_{model_method}_predict_eval_strategy.stats',

        'is_upgrade_model': True,
        'stock_list': [target_symbol, 'DX-Y.NYB', '^VIX',], #'XLE','XLP',]#, 'XLP', 'XLF',]#'GLD']# ]#'^TNX'] #, ]
        #note: add ^VIX bad sharpe vs not. add 'DX-Y.NYB' positive vs not. ADD '^TNX' negative

        'o_forcast_result_file': f'{root_pth}/datas/{target_symbol}'+'_{target_price}_xgb_forcast_result.csv',
        'forcast_model_save': f'{root_pth}/datas/{target_symbol}'+'_{target_price}_forcast_model.json',
    }
    # if target_symbol == 'TQQQ':
    #     param['stock_list'].append('SQQQ')
    # elif target_symbol == 'SQQQ':
    #     param['stock_list'].append('TQQQ')
    # else:
    #     pass

    return param

def e2eBacktest(param, is_classifier = True):
    # today = datetime.datetime.now().strftime('%Y-%m-%d')
    # yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    setLog(param['target'])
    if param['is_realtime']:
        assert param['split_ratio']['train'] == 0.0
        assert param['split_ratio']['dev'] == 0.0
        assert param['split_ratio']['eval'] == 1.0

    #Step-1: read CSV raw data & label samples for model training
    target_symbol = param['target']# 'TQQQ'
    # fl_stock_list = ['../zipline/datas/TQQQ.csv','../zipline/datas/VIX.csv','../zipline/datas/DX-Y.NYB.csv'] #,'datas/TNX.csv']    
    fl_stock_list = param['stock_list'] #[target_symbol,'DX-Y.NYB']#, '^TNX']'^VIX', 
    market_name = 'nasdaq'
    

    cache_file = param['cache_file'] #f'{root_pth}/{target_symbol}_stock_cache.pkl'
    if param['refresh_cache'] or not os.path.exists(cache_file):
        download = tickerDownloader()
        data = download.downloadYahoo(market_name, fl_stock_list, batch_size = 1, interval='1d')

        with open(cache_file, 'wb') as fi:
            pickle.dump(data, fi)
    else:
        try:
            with open(cache_file, 'rb') as fi:
                data = pickle.load(fi)
        except:
            raise Exception(f'*** Warning: cached download {cache_file} not exists')
    
    #add manul latest prices
    if param.get('realtime_price'):
        rv = param.get('realtime_price')
        new_record = {'date': [rv['date']]}
        for v in list(data[target_symbol].columns):
            try:
                new_record[v] = [rv[v]]
            except:
                new_record[v] = [0.0]
        new_record = pd.DataFrame.from_dict(new_record)
        new_record.set_index('date', inplace=True)
        # print(new_record)
        # nid = data[target_symbol].index.append(rv['date'])
        # print(nid.tail(3))
        # data[target_symbol].loc[len(data[target_symbol].index)] = new_record
        # data[target_symbol][rv['date']] = new_record
        new_record = pd.concat([data[target_symbol], new_record], axis = 0)
        # print(new_record.tail(3))
        # df_new = pd.merge(data[target_symbol], latest_price_1d, how = 'outer', left_index=True, right_index=True, suffixes = ['',''])#, on='index')
        # print(df_new.tail(3))
    #     print(latest_price_1d)
    #     for v in list(data[target_symbol].columns):
    #         try:
    #             data[target_symbol].loc[latest_price_1d.index[0]][v] = rv[v]
    #         except:
    #             data[target_symbol].loc[latest_price_1d.index[0]][v] = None
    #     print(data[target_symbol].tail(3))
    # #
    if param.get('is_offline_strategy'):
        for ky, kv in data.items():
            data[ky] = kv[kv.index.isin(pd.date_range(param['first_date'], param['last_date']))]

    #Step-2: Prepare feature
    addon_ticker = list(filter(lambda x:x != target_symbol, fl_stock_list))
    signal = signalClassifier(price_chg_margin = param['price_margin'], label_price_source = f"{param['label_price_source']}_raw", lookback_day = param['lookback_window'], price_norm = param['price_scaler'], n_day_hist_feature=param['n_lookback_feat'])
    data_feat_hist = signal.feature_to_learner(data, target_symbol, addon_ticker, is_realtime = param['is_realtime'], split_ratio=param['split_ratio'])

    #cache feature
    cache_feat_file = param['cache_feat_file'] #f'{root_pth}/{target_symbol}_stock_cache.pkl'
    if cache_feat_file:
        with open(cache_feat_file, 'wb') as fi:
            pickle.dump(data_feat_hist, fi)

    #setp-3: model optimization
    n_stake = param['stake'] #10 #40
    cash_capital = param['cash']

    signal_file =param['o_signal_file'] #
    model_file  = param['model_save'] 

    if param['is_classifier']:
        learn_model = learner()
        if param['is_realtime']:
            signal_tag = {
                0: 'Hold',
                1: 'Buy',
                2: 'sell',
            }

            if param.get('is_offline_strategy'):
                learn_model.modelPredict(target_symbol, data_feat_hist['eval'], data, signal_file,  model_file) 
                #step-4: strategy evaluation
                o_strategy_log_file = param['o_log_file'] #
                o_quant_output_file = param['o_quant_file'] #
                o_quant_output_html = param['o_quant_html'] #
                strategyEvaluateLongshort(target_symbol, signal_file, o_strategy_log_file, o_quant_output_file, o_quant_output_html, n_stake, cash_capital, param['is_onlylong'])
            else:
                result = learn_model.signalPredict(target_symbol, data_feat_hist, data, model_file)
                predicted_signal = result['predict'][0]
                predicted_signal = signal_tag[predicted_signal]  

                today = pd.to_datetime(result.index[-1])#.dt.date #datetime.datetime.now().strftime('%Y-%m-%d')
                tomorrow = (today + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

                print(f'{predicted_signal} on {tomorrow}, predicted on {today}, ')
        else:
            if param['is_upgrade_model']:
                result = learn_model.fit(target_symbol, data_feat_hist, data, signal_file,save_model = model_file, tune_obj_reward = {'n_stake':n_stake, 'cash_capital':cash_capital})
            result2 = learn_model.modelPredict(target_symbol, data_feat_hist['eval'], data, signal_file,  model_file) 

            with open(param['model_acc_report'], 'w') as fi:
                json.dump(result2, fi, indent=6)
                    
            #step-4: strategy evaluation
            o_strategy_log_file = param['o_log_file'] #
            o_quant_output_file = param['o_quant_file'] #
            o_quant_output_html = param['o_quant_html'] #

            strategyEvaluateLongshort(target_symbol, signal_file, o_strategy_log_file, o_quant_output_file, o_quant_output_html, n_stake, cash_capital, param['is_onlylong'])
    else:
        pred_df = None
        for target_price in ('Open', 'Close', 'High', 'Low'):
            save_model = param['forcast_model_save'].format(target_price=target_price, target_symbol=target_symbol)
            learn_model = learnerRegressor(scale_price = 'log', n_lookback_feat = 20)
            if  not param['is_realtime']: 
                if param['is_upgrade_model']:
                    result = learn_model.fit(target_price, data_feat_hist, data[target_symbol], save_model)                    
            else:
                o_forcast_result_file = param['o_forcast_result_file'].format(target_price=target_price, target_symbol=target_symbol)
                next_day = learn_model.predict(target_symbol, target_price, data_feat_hist, data[target_symbol], o_forcast_result_file, save_model)
                next_day.rename(columns = {'predict_price': f'predict_{target_price}'}, inplace=True)
                # print(f'{target_price} = {next_day}')
                if pred_df is not None:
                    pred_df = pd.concat([pred_df, next_day], axis=1)
                else:
                    # pred_df = pd.DataFrame(index = next_day.index)
                    pred_df = next_day
        if pred_df is not None:
            pred_df['predict_middle'] = (0.5 * (pred_df['predict_High'] + pred_df['predict_Low'])).round(4)
            print(pred_df)            

def run(args_me = None):
    try:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='e2e trading function')

        parser.add_argument('--target', required=False,
                            default='TQQQ',
                            help='target symbol to trade & predict')

        parser.add_argument('--fromdate', required=False, default=None,
                            help='Starting date in YYYY-MM-DD format')

        parser.add_argument('--todate', required=False, default=None,
                            help='Ending date in YYYY-MM-DD format')

        parser.add_argument('--func', required=False,
                            default='realtime',
                            help='func = realtime, train, strategy')
        
        parser.add_argument('--model', required=True, default=None,
                            help='model type: MUST xgb, catboost')
        parser.add_argument('--price_margin', required=False, default=0.01, type=float, #1% price change
                            help='price margin: change of price when decide Hold')
        parser.add_argument('--capital', required=False, default=100, type=float, #1% price change
                            help='total capital.')

        #for realtime prediction, manually input latest price    
        parser.add_argument('--high', required=False, default=None,
                            help='latest high price')
        parser.add_argument('--low', required=False, default=None,
                            help='latest low price')
        parser.add_argument('--open', required=False, default=None,
                            help='latest open price')
        parser.add_argument('--close', required=False, default=None,
                            help='latest close price')


        args = parser.parse_args()
        print(args)
    except:
        args = args_me
    
    if args.model not in ('xgb', 'catboost'):
        raise Exception('** warning: --model must be xgb or catboost')
    param = configure(args) #.target, args.model, args.price_margin)

    if args.func in ('strategy',):
        try:
            first_date = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d') #datetime.datetime(2021, 1, 1) 
            last_date = datetime.datetime.strptime(args.todate, '%Y-%m-%d') #datetime.datetime(2022, 9, 30)
        except:
            raise Exception('--fromdate --todate MUST be set for strategy, realtime')

    if args.func == 'realtime':
        param['is_classifier'] = True
        param['is_realtime'] = True
        param['split_ratio'] = {'train': 0.0, 'dev': 0.0, 'eval': 1.0}
        param['price_margin'] = args.price_margin
        e2eBacktest(param)
    elif args.func == 'train':
        param['is_classifier'] = True
        param['is_realtime'] = False
        param['split_ratio'] = {'train': 0.7, 'dev': 0.1, 'eval': 0.2}
        param['price_margin'] = args.price_margin
        e2eBacktest(param)
    elif args.func == 'strategy':
        param['is_classifier'] = True #DO NOT Change
        param['is_realtime'] = True #DO NOT Change
        param['split_ratio'] = {'train': 0.0, 'dev': 0.0, 'eval': 1.0}
        param['first_date'] = first_date
        param['last_date'] = last_date
        param['is_offline_strategy'] = True
        param['price_margin'] = args.price_margin
        e2eBacktest(param)
        pass# backtestStrategy(param, first_date, last_date)
    elif args.func == 'regressor':
        param['is_classifier'] = False
        param['is_realtime'] = False
        param['split_ratio'] = {'train': 0.7, 'dev': 0.1, 'eval': 0.2}
        e2eBacktest(param)
    elif args.func == 'regressor_predict':
        param['is_classifier'] = False
        param['is_realtime'] = True
        param['split_ratio'] = {'train': 0.0, 'dev': 0.0, 'eval':1.0}
        #check if realtime input
        # is_manual_input = 0
        # for v in ['high', 'low', 'close', 'open']:
        #     if v in args:
        #         is_manual_input += 1
        # print(f'**** {is_manual_input}')
        # if is_manual_input == 4:
        #     print('*** INFO: manual update latest price')
        #     param['realtime_price'] = {
        #         'date': datetime.datetime.strptime(str(datetime.date.today()), '%Y-%m-%d'),#datetime.datetime.now(), '%Y-%m-%d'),
        #         'High': float(args.high), 
        #         'Low': float(args.low),
        #         'Close': float(args.close),
        #         'Open': float(args.open),
        #         'Volume': 0.0,
        #         'Symbol': param['target'],
        #         'Stock_splits': 0.0,
        #         'mkt_name': 'nasdaq'
        #     }
            
        # elif is_manual_input > 0 and is_manual_input < 4:
        #     raise Exception("*** Warning: manual input must input 4 args: 'high', 'low', 'close', 'open'")  
        # else:
        #     print('*** INFO: predict based on yesterday')

        e2eBacktest(param)
    elif args.func == 'hold_day':
        pass
    else:
        raise Exception('Warning: func MUST be realtime, train, strategy')


if __name__ == '__main__':
    is_debug = False
    if is_debug:
        args = {}
        args['target'] = 'TSLA'
        args['model'] = 'xgb'
        args['func'] = 'train'
        args['capital'] = 1000
        args['price_margin'] = 0.01

        param = configure(args)
        param['is_classifier'] = True #DO NOT Change
        param['is_realtime'] = False
        param['i_price_margin'] = 0.01
        param['split_ratio'] = {'train': 0.7, 'dev': 0.1, 'eval': 0.2}
        print(param)
        e2eBacktest(param)    
    else:
        run()

