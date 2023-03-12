#*-* coding: utf-8 *-*
#!/usr/bin/env python3
import numpy as np
import math
import pickle
import xgboost as xgb
import datetime
import pandas as pd
from hyperopt import tpe, hp, fmin, Trials
import copy
from .mlStrategyLongshort import strategyEvaluate
import logging
from .melog import infome
from .featureOpt import signalClassifier

MAX_TRIAL_TRAIN = 2

class learner():
    '''
    model training & prediction
    '''

    def __init__(self, **kwarg):
        pass
        # super().__init__(kwarg)
    
    def __acc_metric(self, y1, y2):
        '''
        Calculate accuracy
        '''
        res = list(zip(y1, y2))
        correct = 0
        total = 0
        for v in res:
            total += 1
            if v[0] == v[1]:
                correct += 1

        return correct/float(total)
    def __error_metric(self, y1, y2):
        '''
        Calculate accuracy
        '''
        res = list(filter(lambda x: math.isfinite(x[0]) and math.isfinite(x[1]), zip(y1, y2)))
        correct = 0
        total = 0
        for v in res:
            correct += (v[0]-v[1]) * (v[0]-v[1])

        return math.sqrt(correct/float(len(res)))

    def __balanceLabel(self, i_df):
        for v in i_df.columns:
            if v != 'label':
                col = v
                break
        n_count_label = i_df.groupby(by='label').count()[col]
        try:
            n_sample = min(n_count_label[1], n_count_label[2])
            o_df = i_df[i_df['label']==0]
            # n_sample = min(n_count_label)
            # o_df = i_df[i_df['label']==0].sample(n_sample, random_state=20220930)
            p_df = i_df[i_df['label']==1].sample(n_sample, random_state=20220930)
            n_df = i_df[i_df['label']==2].sample(n_sample,  random_state=20220930)
            return pd.concat([o_df,p_df,n_df], axis = 0), False
        except:
            n_sample = min(n_count_label[1], n_count_label[0])
            o_df = i_df[i_df['label']==0].sample(n_sample, random_state=20220930)
            p_df = i_df[i_df['label']==1].sample(n_sample, random_state=20220930)
            return pd.concat([o_df,p_df], axis = 0), True


    def fit(self, target_symbol, train_dev_eval_set, data_raw, o_prediction_file, save_model = './xgb_model.json', tune_obj_reward = {'n_stake':1, 'cash_capital':1000}):
        '''
        Desc: tune ML model

        train_dev_eval_set: input train/dev/test data set

        '''
        tuning_report = {}

        #step-1: model training
        #prepare training data
        train_dev_eval_set['train'], is_binary = self.__balanceLabel(train_dev_eval_set['train'])
        # train_dev_eval_set['dev'],is_binary = balanceLabel(train_dev_eval_set['dev'])

        feat_cols = list(filter(lambda x: x != 'label', train_dev_eval_set['train'].columns))
        label_cols = 'label'

        x_train = train_dev_eval_set['train'][feat_cols].to_numpy()
        y_train = train_dev_eval_set['train'][label_cols].to_numpy().astype(int)

        x_dev = train_dev_eval_set['dev'][feat_cols].to_numpy()
        y_dev = train_dev_eval_set['dev'][label_cols].to_numpy().astype(int)

        eval_set_df = train_dev_eval_set['dev']
        # o_prediction_file = './opt_model_dev.csv'

        if is_binary:
            objective = 'binary:logistic'
            eval_metric = 'error'
        else:
            objective = 'multi:softprob'
            eval_metric = 'merror'
        
        base_xgb_cfg = {
            'n_estimators': 0,
            'objective' : objective,
            'booster' : 'gbtree', #'gbtree'
            'eta': 0.01,
            'reg_lambda' : 0,
        #     reg_alpha = 10,
            'max_depth': 0,
            'verbosity': 0,
            'eval_metric' : eval_metric, #'auc', #'merror', #'mlogloss',
            'nthread': 3,
            'rate_drop':0.,
            'subsample': 1.0,

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
            model_now = xgb.XGBClassifier(**xgb_cfg)
            model_now.fit(x_train, y_train, eval_set = [(x_dev, y_dev)])#, early_stopping_rounds=50, verbose=True)
            # y_dev_pred = model_now.predict(x_dev)
            dev_result = model_now.evals_result()
            dev_acc = dev_result['validation_0'][eval_metric][0]
            # dev_acc = acc_metric(y_dev_pred, y_dev)

            #model predict in dev
            feat_cols = list(filter(lambda x: x != 'label', eval_set_df.columns))
            label_cols = 'label' 
            #evaluate set
            x_eval = eval_set_df[feat_cols].to_numpy()
            y_eval = eval_set_df[label_cols].to_numpy().astype(int)
            y_eval_pred = model_now.predict(x_eval)
            y_eval_pred_df = pd.DataFrame(eval_set_df['label'])
            y_eval_pred_df['predict'] = y_eval_pred
            #Save prediction 
            o_cols = ['Open', 'High', 'Low','Close', 'Volume',]
            
            o_test = data_raw[target_symbol].loc[y_eval_pred_df.index][o_cols]
            o_test['openinterest'] = 0
            o_test = pd.merge(o_test, y_eval_pred_df['predict'], on='Date', how='inner')
            o_test.to_csv(o_prediction_file, index = True)
            reward_gain = strategyEvaluate(target_symbol, o_prediction_file, n_stake=tune_obj_reward['n_stake'], cash_capital= tune_obj_reward['cash_capital'], is_model_opt=True)

            return -reward_gain #dev_acc #

        search_space = {
            'n_estimators': hp.randint('n_estimators', 5, 50),
            'eta': hp.uniform('eta', 0.001, 0.01),
            'max_depth': hp.randint('max_depth', 2, 8),
            'reg_lambda': hp.randint('reg_lambda', 1, 10),
        }
        best_model = None
        best_model_val = None
        n_cycle = 0
        while(True):
            opt_trials = Trials()
            best = fmin(
                fn = model_cost,
                space = search_space,
                algo = tpe.suggest,
                max_evals = MAX_TRIAL_TRAIN,
                trials=opt_trials,        
            )
            logging.info(infome(__file__, '*** XGB MODEL tuinig *** '))
            best_value = sorted(opt_trials.results, key = lambda v: v['loss'])
            logging.info(infome(__file__, f'best = {best}, {best_value[0]} '))

            if best_model:
                if best_value[0]['loss'] < best_model_val['loss']:
                    best_model_val = best_value[0]
                    best_model = best
            else:
                best_model_val = best_value[0]
                best_model = best

            print(f'**** This is {n_cycle}-th try, {best}')
            n_cycle += 1
            if n_cycle >= 6:# and best_model_val['loss'] < -1.0: 
                break

        xgb_cfg = copy.deepcopy(base_xgb_cfg)
        xgb_cfg.update(
            {
            'n_estimators': best_model['n_estimators'],
            'eta': best_model['eta'],
            'reg_lambda' : best_model['reg_lambda'],
            'max_depth': best_model['max_depth'],
        }
        )
        print(xgb_cfg)
        model_now = xgb.XGBClassifier(**xgb_cfg)
        model_now.fit(x_train, y_train, eval_set = [(x_dev, y_dev)])#, early_stopping_rounds=50, verbose=True)
    
        model_now.save_model(save_model)
        
        # #report
        tuning_report['model'] = 'xgb'
        tuning_report['param'] = xgb_cfg

        #evaluate model
        model_now.load_model(save_model)
        y_train_pred = model_now.predict(x_train)
        acc = self.__acc_metric(y_train_pred, y_train)
        print(f'train = {acc}')
        tuning_report['metric'] = {'train': acc}
        
        y_dev_pred = model_now.predict(x_dev)
        dev_acc = self.__acc_metric(y_dev_pred, y_dev)
        print(f'dev = {dev_acc}')
        tuning_report['metric'].update({'dev': dev_acc})
    
        return tuning_report

    def modelPredict(self, target_symbol, eval_set, data_raw, o_prediction_file, save_model):
        '''
        Desc: tune ML model

        train_dev_eval_set: input train/dev/test data set

        '''
        model_now = xgb.XGBClassifier() #**xgb_cfg)
        model_now.load_model(save_model)

        tuning_report = {}

        feat_cols = list(filter(lambda x: x != 'label', eval_set.columns))
        label_cols = 'label' 
        #evaluate set
        x_eval = eval_set[feat_cols].to_numpy()
        y_eval = eval_set[label_cols].to_numpy().astype(int)
        y_eval_pred = model_now.predict(x_eval)
        eval_acc = self.__acc_metric(y_eval_pred, y_eval)
        print(f'eval = {eval_acc}') 
        y_eval_pred_df = pd.DataFrame(eval_set['label'])
        y_eval_pred_df['predict'] = y_eval_pred
        tuning_report['metric'] = {'eval': eval_acc}

        #Save prediction 
        o_cols = ['Open', 'High', 'Low','Close',  'Volume',]
        
        o_test = data_raw[target_symbol].loc[y_eval_pred_df.index][o_cols]
        yr1, mth1, day1 = list(map(lambda x:int(x),str(o_test.index[0]).split(' ')[0].split('-')))
        yr2, mth2, day2 = list(map(lambda x:int(x),str(o_test.index[-1]).split(' ')[0].split('-')))
        o_test['openinterest'] = 0
        o_test = pd.merge(o_test, y_eval_pred_df['predict'], on='Date', how='inner')
        o_test.to_csv(o_prediction_file, index = True)
        print(f'{yr1}, {mth1}, {day1}')

        tuning_report['save_prediction'] = o_prediction_file
        #

        return tuning_report

    def signalPredict(self, target_symbol, eval_set, data_raw, save_model):
        '''
        Desc: tune ML model

        train_dev_eval_set: input train/dev/test data set

        '''
        tuning_report = {}

        #step-1: model training
        #prepare training data
        feat_cols = list(filter(lambda x: x != 'label', eval_set['eval'].columns))
        #evaluate model
        model_now = xgb.XGBClassifier()
        model_now.load_model(save_model)
        latest_ds_data = eval_set['eval'][feat_cols].tail(1)
        x_eval = latest_ds_data.to_numpy()
        # y_eval = latest_ds_data.to_numpy().astype(int)
        y_eval_pred = model_now.predict(x_eval)

        #Save prediction 
        o_cols = ['Open', 'High', 'Low','Close', 'Volume',]
        
        o_test = data_raw[target_symbol].loc[latest_ds_data.index][o_cols]
        o_test['openinterest'] = 0
        o_test['predict'] = y_eval_pred

        return o_test

if __name__ == '__main__':
    pass





