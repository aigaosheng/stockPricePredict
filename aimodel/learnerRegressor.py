#*-* coding: utf-8 *-*
#!/usr/bin/env python3
import numpy as np
import math
import pickle
import xgboost as xgb
from catboost import CatBoostClassifier, Pool, cv
import datetime
import pandas as pd
from hyperopt import tpe, hp, fmin, Trials
import copy
from .mlStrategyLongshort import strategyEvaluate
import logging
from .melog import infome

class learnerRegressor():
    def __init__(self, scale_price = 'log', n_lookback_feat = 3):
        self.price_scale_method = scale_price
        if self.price_scale_method == 'log':
            self.fnc = np.log
            self.fnc_inv = np.exp
        elif self.price_scale_method == 'sqrt':
            self.fnc = np.sqrt
            self.fnc_inv = lambda x: x*x
        else:
            self.fnc = lambda x: x
            self.fnc_inv = lambda x: x
        
        self.n_lookback_feat = n_lookback_feat

    def forcastFeature(self, target_price, data_raw, i_feat, is_realtime = False):
        '''
        Arguments: 
        target_price: target price column
        data_raw: raw data
        i_feat: feature vector prepred for classifier
        '''
        cols = list(data_raw.columns)
        if target_price not in cols:
            raise Exception(f'*** Warning: {target_price} not in {cols}')
        dfeat = {}
        dft = pd.DataFrame(index = data_raw.index)
        dft[target_price] = data_raw[target_price].apply(self.fnc)

        feat_cols = [target_price]
        for k in range(1, self.n_lookback_feat + 1):
            ky = f'{target_price}_{k}'
            dft[ky] = dft[target_price].shift(k).sub(dft[target_price].shift(k-1)) 
            feat_cols.append(ky)
            ky2 = f'dif_{k}'
            dft[ky2] = dft[ky] - dft[ky].shift(1)
            feat_cols.append(ky2)

        if is_realtime:
            dft['label'] = 0
        else:    
            dft['label'] = data_raw[target_price].shift(-1).div(data_raw[target_price]) - 1

        feat_cols.append('label')
        dft = dft[feat_cols]
        #split
        if not is_realtime:
            dft.dropna(inplace=True)
            dft_model_train = {} #dataSplit(dft, split_ratio = {'train': 0.7, 'dev': 0.1, 'eval': 0.2}, filter_feature_cols=[])
            for ky, kv in i_feat.items():
                dft_model_train[ky] = dft.loc[kv.index.intersection(dft.index)]
                dft_model_train[ky] = dft_model_train[ky].merge(kv, how='inner', left_index=True, right_index=True, suffixes=('','_y'))
        else:
            dft_model_train = {'eval': dft}
            dft_model_train['eval'] = dft_model_train['eval'].merge(i_feat['eval'], how='inner', left_index=True, right_index=True, suffixes=('', '_y'))

        return dft_model_train        

    def mapRatio2Price(self, ratio, today_price):
        pred_price = []
        for k, v in zip(ratio, today_price):
            if self.price_scale_method == 'log':
                v = np.exp(v) * (1+k)
            elif self.price_scale_method == 'sqrt':
                v = (1+k) * (v*v)
            else:
                v = (1+k)*v
            pred_price.append(v)
        return pred_price

    # @staticmethod
    def error_metric(self, y1, y2):
        '''
        Calculate accuracy
        '''
        res = list(filter(lambda x: math.isfinite(x[0]) and math.isfinite(x[1]), zip(y1, y2)))
        correct = 0
        total = 0
        for v in res:
            correct += (v[0]-v[1]) * (v[0]-v[1])

        return math.sqrt(correct/float(len(res)))


    def fit(self, target_price, i_train_dev_eval_set, data_raw, save_model):
        '''
        Desc: tune ML model

        train_dev_eval_set: input train/dev/test data set

        '''
        train_dev_eval_set = self.forcastFeature(target_price, data_raw, i_train_dev_eval_set, is_realtime = False)

        tuning_report = {}

        #step-1: model training

        feat_cols = list(filter(lambda x: x not in ('label',target_price), train_dev_eval_set['train'].columns))
        label_cols = 'label'

        x_train = train_dev_eval_set['train'][feat_cols].to_numpy()
        y_train = train_dev_eval_set['train'][label_cols].to_numpy()#.astype(int)

        x_dev = train_dev_eval_set['dev'][feat_cols].to_numpy()
        y_dev = train_dev_eval_set['dev'][label_cols].to_numpy()#.astype(int)

        eval_metric_me = 'mae' #eval_metric_me
        base_xgb_cfg = {
            'n_estimators': 20,
            'objective' : 'reg:squarederror',#'reg:squarederror',
            # 'booster' : 'gbtree', #'gbtree'
            'eta': 0.02,
            'reg_lambda' : 1,
        #     reg_alpha = 10,
            'max_depth': 6,
            'verbosity':0,
            'eval_metric' : eval_metric_me,# eval_metric_me, #'mlogloss',
            'nthread': 3,
            'subsample': 1.0,
            'base_score': 0,
            'tree_method': 'hist',

        }

        def model_cost(model_cfg):
            xgb_cfg = copy.deepcopy(base_xgb_cfg)
            xgb_cfg.update(
                {
                'n_estimators': model_cfg['n_estimators'],
                'eta': model_cfg['eta'],
                'reg_lambda' : model_cfg['reg_lambda'],
                'max_depth': model_cfg['max_depth'],
            }
            )
            # print(xgb_cfg)
            model_now = xgb.XGBRFRegressor(**xgb_cfg)
            model_now.fit(x_train, y_train, eval_set = [(x_dev, y_dev)], verbose=True)
            dev_result = model_now.evals_result()
            v = dev_result['validation_0'][eval_metric_me][0]
            return v

        search_space = {
            'n_estimators': hp.randint('n_estimators', 2, 50),
            'eta': hp.uniform('eta', 0.1, 1),
            'max_depth': hp.randint('max_depth', 2, 8),
            'reg_lambda': hp.randint('reg_lambda', 1, 5),
        }
        best = fmin(
            fn = model_cost,
            space = search_space,
            algo = tpe.suggest,
            max_evals = 10
        )
        print(best)
        # best = base_xgb_cfg
        
        xgb_cfg = copy.deepcopy(base_xgb_cfg)
        xgb_cfg.update(
            {
            'n_estimators': best['n_estimators'],
            'eta': best['eta'],
            'reg_lambda' : best['reg_lambda'],
            'max_depth': best['max_depth'],
        }
        )
        print(xgb_cfg)

        model_now = xgb.XGBRFRegressor(**xgb_cfg)
        model_now.fit(x_train, y_train, eval_set = [(x_dev, y_dev)], verbose=True)
        dev_result=model_now.evals_result()
        a = dev_result['validation_0'][eval_metric_me][0]
        # print(a)
        model_now.save_model(save_model)
        
        # #report
        tuning_report['model'] = 'xgb'
        tuning_report['param'] = xgb_cfg

        #evaluate model
        model_now.load_model(save_model)
        y_train_pred = model_now.predict(x_train)
        #map to price
        lst_train = list(train_dev_eval_set['train'][target_price])
        lst_train_true = list(train_dev_eval_set['train'][target_price].apply(self.fnc_inv).shift(-1)) #next day
        y_train_pred2 = self.mapRatio2Price(y_train_pred, lst_train)
        #
        acc_r = self.error_metric(y_train_pred, y_train) #predicted ratio error
        acc = self.error_metric(y_train_pred2, lst_train_true) #predicted price error
        print(f'train = {acc}, {acc_r}')
        tuning_report['metric'] = {'train': acc}   

        y_dev_pred = model_now.predict(x_dev)
        lst_dev_true = list(train_dev_eval_set['dev'][target_price].apply(self.fnc_inv).shift(-1))
        lst_dev = list(train_dev_eval_set['dev'][target_price])
        y_dev_pred2 = self.mapRatio2Price(y_dev_pred, lst_dev)

        # print(y_dev_pred2)
        acc_r = self.error_metric(y_dev_pred, y_dev)
        dev_acc = self.error_metric(y_dev_pred2, lst_dev_true)
        print(f'dev = {dev_acc}, {acc_r}')
        tuning_report['metric'].update({'dev': dev_acc})
        
        dev1 = list(zip(y_dev_pred, y_dev))
        # print(dev1[:5])

        x_eval = train_dev_eval_set['eval'][feat_cols].to_numpy()
        y_eval = train_dev_eval_set['eval'][label_cols].to_numpy()#.astype(int)
        
        y_eval_pred = model_now.predict(x_eval)
        
        lst_eval_true = list(train_dev_eval_set['eval'][target_price].apply(self.fnc_inv).shift(-1))   
        lst_eval = list(train_dev_eval_set['eval'][target_price])
        y_eval_pred2 = self.mapRatio2Price(y_eval_pred, lst_eval)
        
        dev_acc_r = self.error_metric(y_eval_pred, y_eval)
        dev_acc = self.error_metric(y_eval_pred2, lst_eval_true)
        print(f'dev = {dev_acc}, {dev_acc_r}')
        tuning_report['metric'].update({'eval': dev_acc})


        return tuning_report,dev1,list(zip(y_train_pred2, lst_train)),list(zip(y_dev_pred2, lst_dev)) ,list(zip(y_eval_pred2, lst_eval))      


    def predict(self, target_symbol, target_price, i_train_dev_eval_set, data_raw, o_prediction_file, save_model = './xgb_model_forcast.json'):
        '''
        Desc: tune ML model

        train_dev_eval_set: input train/dev/test data set

        '''
        train_dev_eval_set = self.forcastFeature(target_price, data_raw, i_train_dev_eval_set, is_realtime = True)

        tuning_report = {}

        #step-1: model training
        #prepare training data

        feat_cols = list(filter(lambda x: x not in ('label',target_price), train_dev_eval_set['eval'].columns))
        label_cols = 'label'

        eval_set_df = train_dev_eval_set['eval']

        x_eval = train_dev_eval_set['eval'][feat_cols].to_numpy()
        y_eval = train_dev_eval_set['eval'][label_cols].to_numpy()#.astype(int)

        #evaluate model
        model_now = xgb.XGBRFRegressor()
        model_now.load_model(save_model)
        
        y_eval_pred = model_now.predict(x_eval)
        
        lst_eval = list(train_dev_eval_set['eval'][target_price])  #today price 
        y_eval_pred2 = self.mapRatio2Price(y_eval_pred, lst_eval)

        eval_set_df = train_dev_eval_set['eval']
        feat_cols = list(filter(lambda x: x != 'label', eval_set_df.columns))
        label_cols = 'label' 
        #evaluate set
        y_eval_pred_df = pd.DataFrame(eval_set_df['label'])
        y_eval_pred_df['predict'] = y_eval_pred

        y_eval_pred_df['target_price'] = eval_set_df[target_price].apply(self.fnc_inv)
        y_eval_pred_df['predict_price'] = y_eval_pred2

        next_day_pred = y_eval_pred2[-1]
        y_eval_pred_df['predict_price'] = y_eval_pred_df['predict_price'].shift(1)
        #Save prediction 
        o_cols = ['Open', 'High', 'Low','Close',  'Volume',]
        
        o_test = data_raw.loc[y_eval_pred_df.index][o_cols]
        o_test['openinterest'] = 0
        o_test = pd.merge(o_test, y_eval_pred_df[['predict','target_price','predict_price']], on='Date', how='inner').round(4)

        #patch tomorrow
        nex_day = pd.DataFrame(index=y_eval_pred_df.index[-1:] + pd.DateOffset(1), columns = o_test.columns)
        nex_day['predict_price'] = next_day_pred
        o_test = pd.concat([o_test, nex_day], axis=0)
        
        o_test.to_csv(o_prediction_file, index = True)
        
        return o_test.filter(['predict_price']).tail(2).round(4) 

    # def patchExtraFeature(dfeat, data_feature):
    #     cols = list(filter(lambda x:x not in ('reward','reward_mid', 'label'),data_feature['eval'].columns))
    #     for ky,kv in dfeat.items():
    #         dfeat[ky] = pd.concat([dfeat[ky], data_feature[ky][cols]], axis=1, join="inner")
    #     return dfeat

if __name__ == '__main__':
    #read CSV raw data
    cache_raw_file = '/home/gs/Work/fintek/aimodel/TQQQ_stock_cache.pkl'
    with open(cache_raw_file, 'rb') as fi:
        data_raw = pickle.load(fi)
    
    cache_file = '/home/gs/Work/fintek/aimodel/TQQQ_stock_cache_feature.pkl'
    with open(cache_file, 'rb') as fi:
        data_feature = pickle.load(fi)

    # save_model_file = 'TQQQ_{price}_xgb_model_forcast.json'
    # pred_df = None
    # for target_price in ('Open', 'Close', 'High', 'Low'):
    #     save_model = save_model_file.format(price=target_price)
    #     dfeat = forcastFeature('TQQQ', target_price, data_raw['TQQQ'], n_lookback_feat = 3, is_realtime = False)
    #     #
    #     # dfeat = patchExtraFeature(dfeat, data_feature)
    #     #
    #     result2 = fit('TQQQ', target_price, dfeat, data_raw, o_prediction_file=None, save_model = save_model)
        
    #     # print(result2[-1][-10:])

    #     dfeat = forcastFeature('TQQQ', target_price, data_raw['TQQQ'], n_lookback_feat = 3, is_realtime = True)
    #     # dfeat = patchExtraFeature(dfeat, data_feature)

    #     next_day = priceForcast('TQQQ', target_price, dfeat, data_raw, f'./forcast_{target_price}.csv', save_model, is_realtime=True)   
    #     print(f'{target_price}, {next_day}') 
    #     if pred_df is not None:
    #         pred_df[target_price] = next_day
    #     else:
    #         pred_df = pd.DataFrame(index = next_day.index)
    #         pred_df[target_price] = next_day
    # print(pred_df)