#*-* coding: utf-8 *-*
#!/usr/bin/env python3

import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
root_pth = str(Path(__file__).parent.parent.absolute())
# print(root_pth)
sys.path.append(root_pth)#
# print(sys.path)
# sys.path.append('/home/gsw/work/fintek')
from db.dbManager import dbManager
import pickle 
import psycopg2
from dataGetter.yfinance import tickers
import math
from tqdm import tqdm, tnrange
from datetime import datetime,timedelta
import logging

class tickerDownloader():
    def __init__(self):
        self.__postfix_yahoo = {
            'sse': '.SS', 
            'sz': '.SZ',
            'nyse': '', 
            'nasdaq': '' 
        }  
    
    def downloadYahoo(self, market_name, tick_list, batch_size = 1, interval='1d'):
        '''
        desc: download tickers
        Arguments: 
        market_name: nasdaq, nyse, sse, sz
        market_ticker_list: list of tickers, e.g ('AAPL', 'MSFT')
        to download AAPL, MSFT of nasdaq
        batch_size: parallel downloader

        return: key-value dict, key = ticker, value = price data
        '''
        # assert market_name in ('nasdaq','nyse','sse','sz')
        df_hist_price = {}    
        end_ds = datetime.now().strftime('%Y-%m-%d')
        postfix = self.__postfix_yahoo[market_name]
        #
        ticker_download = []
        ticker_download_ds = []
        n_ticker = len(tick_list)
        tick_list = dict(map(lambda x: (x + postfix, None), tick_list))
        pbar = tqdm(total = n_ticker)

        n_block = math.ceil(len(tick_list) / batch_size)
        for ticker_now, v in tick_list.items():
            start_ds = None
            dowload = tickers.Tickers([ticker_now])
            if start_ds is None or start_ds < end_ds:
                hst = dowload.history(period=None,interval = interval, start=start_ds, end = end_ds, group_by=None, progress = False)
                if hst.empty:
                    continue
                try:
                    hst['Stock_splits'] = hst['Stock Splits']
                except:
                    hst['Stock_splits'] = None
                hst['mkt_name'] = market_name
                hst.reset_index(level=0, inplace=True)
                hst.set_index('Date', inplace= True)

                df_hist_price[ticker_now] = hst
            pbar.update(1)
        print(f'*** INFO: {market_name} done. Total tickles = {n_ticker}')
        
        return df_hist_price

if __name__ == '__main__':
    # [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
    download = tickerDownloader()
    market_name = 'nasdaq' #nyse, sse, sz
    market_ticker_list = ['TQQQ', 'DX-Y.NYB', '^VIX',]#'XLE','XLP',]#, 'XLP', 'XLF',]#'GLD']# ]#'^TNX'] #, ]
    df = download.downloadYahoo(market_name, market_ticker_list, batch_size = 1, interval='1d')
    #
    cache_file = '/home/gs/Work/fintek/aimodel/dev/raw_data.pkl'
    with open(cache_file, 'wb') as fi:
        pickle.dump(df, fi)

    # print(df[market_ticker_list[0]].head(10))
