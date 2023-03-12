#*-* coding: utf-8 *-*
#!/usr/bin/env python3
import numpy as np
import math
import pickle
import pandas as pd

class signalClassifier(object):
    '''
    Prepare prediction/regression features for model training & prediction

    argument:
    '''
    def __init__(self, price_chg_margin = 0.01, label_price_source = 'Close_raw', lookback_day = (5, 10, 14, 20,), price_norm = 'log', n_day_hist_feature=3):
        '''
        argument: i_price_df, dict of raw price, key = ticker
        target_ticker: ticker name to be predicted
        addon_ticker: ticker list to be used as extra-feature when doing prediction
        price_norm:  
            + log: log price
            + sqrt: sqrt price
            + None: raw price 
        '''
        self.__lookback_day = lookback_day
        self.__price_norm = price_norm
        if self.__price_norm == 'log':
            self.scaler = np.log
            self.scaler_inv = np.exp
        elif self.__price_norm == 'sqrt':
            self.scaler = np.sqrt
            self.scaler_inv = lambda x: x * x
        else:
            self.scaler = lambda x: x
            self.scaler_inv = lambda x: x

        self.__price_chg_margin = price_chg_margin 
        self.__label_price_source = label_price_source
        self.__n_day_hist_feature = n_day_hist_feature

        self.__price_cols = {'High': 'High_raw', 'Low': 'Low_raw', 'Open': 'Open_raw', 'Close': 'Close_raw', 'mid': 'mid_raw'}

    def feature(self, i_price_df, target_ticker, addon_ticker = None, is_realtime = False):
        '''
        feture for 
        '''
        #raw price transform
        ticker_ready = i_price_df.keys()
        assert target_ticker in ticker_ready, f'Ticker = {target_ticker} not in input dataframe dict'
        if addon_ticker:
            for v in addon_ticker:
                assert v in ticker_ready, f'Ticker = {v} not in input dataframe dict'

        self.__target_ticker = target_ticker
        if not isinstance(addon_ticker, list):
            addon_ticker = [addon_ticker]
        self.__addon_ticker = addon_ticker

        self.__raw_price = {}
        for ky, kv in i_price_df.items():
            a = pd.DataFrame(kv, columns=self.__price_cols.keys())
            a['mid'] = (kv['High'] + kv['Low']) * 0.5
            b = a.apply(self.scaler)
            self.__raw_price[ky] = pd.concat([a.rename(columns=self.__price_cols), b], axis=1)

        #extract statistics as feature
        self.feature = {}
        ticker_list = [self.__target_ticker]
        if self.__addon_ticker:
            ticker_list.extend(self.__addon_ticker)
        
        for ky in ticker_list:
            self.feature[ky] = self.__sigleTickerFeature(self.__raw_price[ky])

        #get label
        self.feature_label = self.__getLabel(self.feature[self.__target_ticker], is_realtime)
        col_used = []
        for v in self.feature[self.__target_ticker].columns:
            is_use = True
            for c in list(self.__price_cols.keys()) + list(self.__price_cols.values()):
                if v.startswith(c):
                    is_use = False
                    break
            if is_use:
                col_used.append(v)
        self.feature_label = pd.concat([self.feature_label, self.feature[self.__target_ticker][col_used]], axis=1)

        #concatenate addon if used.
        if self.__addon_ticker:
            for ky in self.__addon_ticker:
                self.feature_label = pd.merge(self.feature_label, self.feature[ky][col_used], how='left', left_index=True, right_index=True, suffixes=('',f'_{ky}'))
                if ky == '^VIX':
                    self.feature_label['Close_^VIX'] = self.feature[ky][['Close']]

        return self.feature_label

    def __sigleTickerFeature(self, df):
        '''
        Extract stats of security price sequence as feature 
        df: stock price dataframe, MUST have High, Close, Low, Open
        lookback_day: list of periods to extract feature stats
        return: feature stat dataFrame 
        '''
        assert 'High' in df.columns, 'Warning: High must inside'
        assert 'Low' in df.columns, 'Warning: Low must inside'
        assert 'Open' in df.columns, 'Warning: Open must inside'
        assert 'Close' in df.columns, 'Warning: Close must inside'

        #base price stat sequence
        base_seq = []

        #Today vs yesterday, pnt of price change
        price_tmp = df['Close_raw']
        df['adj_c_chg'] = price_tmp.div(price_tmp.shift(1)) - 1.0
        base_seq.append('adj_c_chg')
        price_tmp =  df['Close']
        df['adj_cnorm_chg'] = price_tmp.div(price_tmp.shift(1)) - 1.0
        base_seq.append('adj_cnorm_chg')

        #Today vs yesterday, reward change
        price_tmp = df['mid_raw']
        df['adj_m_chg'] = price_tmp.div(price_tmp.shift(1)) - 1.0
        base_seq.append('adj_m_chg')
        price_tmp = df['mid']
        df['adj_mnorm_chg'] = price_tmp.div(price_tmp.shift(1)) - 1.0
        base_seq.append('adj_mnorm_chg')

        #intra-day high-low price change 
        df['intraday_hl_chg'] = df['High_raw'].div(df['Low_raw']) - 1.0
        base_seq.append('intraday_hl_chg')
        df['intraday_hlnorm_chg'] = df['High'].div(df['Low']) - 1.0
        base_seq.append('intraday_hlnorm_chg')

        #intra-day high-low price change 
        df['intraday_oc_chg'] = df['Close_raw'].div(df['Open_raw']) - 1.0
        base_seq.append('intraday_oc_chg')
        df['intraday_ocnorm_chg'] = df['Close'].div(df['Open']) - 1.0
        base_seq.append('intraday_ocnorm_chg')

        #adjacent-day Yesterday close vs today open price change 
        df['adjaday_oc_chg'] = df['Open_raw'].div(df['Close_raw'].shift(1)) - 1.0
        base_seq.append('adjaday_oc_chg')
        df['adjaday_ocnorm_chg'] = df['Open'].div(df['Close'].shift(1)) - 1.0
        base_seq.append('adjaday_ocnorm_chg')

        #mean and variation based normalized value 
        for price_col in base_seq:
            for period in self.__lookback_day:
                ky = f'zscore_{price_col}_{period}'
                df[ky] = (df[price_col]-df[price_col].rolling(period).mean()).div(df[price_col].rolling(period).std(ddof=0))
                
        return df

    def __getLabel(self, data, is_realtime):
        #prepare label = 1:up, 2:down,0:hold
        data = data.dropna()
        # label = pd.DataFrame(index = data.index)
        dtmp = data.filter([self.__label_price_source])
        dtmp['next'] = dtmp.shift(-1) 
        if not is_realtime:
            dtmp.dropna(inplace=True)
        dtmp['up_threshold'] = dtmp[self.__label_price_source] * (1.0 + self.__price_chg_margin)
        dtmp['down_threshold'] = dtmp[self.__label_price_source] * (1.0 - self.__price_chg_margin)
        #look-forward window
        dtmp['up_signal'] = dtmp['next'].gt(dtmp['up_threshold']) * 1
        dtmp['down_signal'] = dtmp['next'].le(dtmp['down_threshold']) * 2
        # up_signal = up_signal.add(down_signal)
        dtmp['label'] = dtmp['up_signal']+dtmp['down_signal']

        return dtmp.filter(['label'])

    def dataSplit(self, feat_data, split_ratio = {'train': 0.7, 'dev': 0.1, 'eval': 0.2}, filter_feature_cols = None):
        '''
        Desc: split input data samples into train, development, & test. The split is based on time, not randomly like doing in general ML
        feat_data: dataFrame of labelled samples
        split_ratio: train/dev/test ratio, KEY fixed
        filter_feature_cols: columns not used in model training

        return: DICT with key-value. Key = train, dev, test

        '''
        if filter_feature_cols and isinstance(filter_feature_cols, list):
            cols_feature = list(filter(lambda x: x not in filter_feature_cols, feat_data.columns))
        else:
            cols_feature = feat_data.columns

        label_samples = feat_data.dropna()
        total_samples = label_samples.shape[0]

        #split data
        train_dev_eval_set = {}
        n_pt = 0
        for ky in ('train', 'dev', 'eval'): #order cannot change
            n_sample_now = n_pt + int(split_ratio[ky] * total_samples)
            if ky == 'eval':
                n_sample_now = total_samples
            train_dev_eval_set[ky] = label_samples[n_pt:n_sample_now]
            n_pt = n_sample_now

        return train_dev_eval_set

    def patchHistFeature(self, i_data_split):
        '''
        patch n-day lookback into feature
        i_data_split: train/dev/eval data
        n_day_hist_feature: patch n-day lookback as feature

        return: feature updated with history
        '''
        o_feature = {}
        for ky, i_data in i_data_split.items():
            cols_feature = list(filter(lambda x: not x.startswith('label'), i_data.columns))

            extra_feat_data = None
            for k in range(2, self.__n_day_hist_feature + 1):
                feat_df = i_data.shift(k).filter(items=cols_feature)
                feat_df.dropna(inplace = True)

                if extra_feat_data is None:
                    extra_feat_data = feat_df
                else:
                    extra_feat_data = pd.merge(extra_feat_data, feat_df, how='inner', left_index=True, right_index=True, suffixes=('',f'_y{k}'))                    
            if extra_feat_data is None:
                df = i_data
            else:
                df = pd.merge(i_data, extra_feat_data, how='inner', left_index=True, right_index=True, suffixes=('',f'_extra'))  
            df = df.dropna()

            o_feature[ky] = df

        return o_feature    

    def feature_to_learner(self, df, target_ticker, addon_ticker, is_realtime = True, split_ratio={'train': 0.7, 'dev': 0.1, 'eval': 0.2}):
        feat = self.feature(df, target_ticker, addon_ticker, is_realtime = True)
        split_feat = self.dataSplit(feat, split_ratio)
        feat4predict = self.patchHistFeature(split_feat)

        return feat4predict



if __name__ == '__main__':
    cache_file = '/home/gs/Work/fintek/aimodel/dev/raw_data.pkl'
    with open(cache_file, 'rb') as fi:
        df = pickle.load(fi)
    target_ticker = 'TQQQ' 
    addon_ticker = ['DX-Y.NYB', '^VIX']

    signal = signalClassifier(price_chg_margin = 0.01, label_price_source = 'Close_raw', lookback_day = (5, 10, 14, 20,), price_norm = 'log', n_day_hist_feature=3)

    feat = signal.feature_to_learner(df, target_ticker, addon_ticker, is_realtime = False, split_ratio={'train': 0., 'dev': 0., 'eval': 1})

    # feat = signal.feature(df, target_ticker, addon_ticker, is_realtime = True)
    # split_feat = signal.dataSplit(feat, split_ratio={'train': 0.7, 'dev': 0.1, 'eval': 0.2})
    # feat4predict = signal.patchHistFeature(split_fea)
    print(feat.head(10))