#-*- coding: utf-8 -*-
import sys
sys.path.append('/home/gs/Work/fintek/dataGetter')
import pandas as pd
from datetime import datetime,timedelta
import psycopg2
import pickle
import numpy as np

class dbManager:
    def __init__(self, user, dbname):
        self.__connect__ = None
        self.__cursor__ = None
        self.__user__ = user
        self.__db__ = dbname
        try:
            self.__connect__ = psycopg2.connect(f"dbname={dbname} user={user}")
            self.__cursor__ = self.__connect__ .cursor()
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Warning: error while connecting to PostgreSQL -> {error}")
        # finally:
        #     if self.__connect__:
        #         self.__connect__.close()
        #         self.__cursor__.close()                    
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # if exc_type:
        #     try:
        #         pass#print("Warning: {}".format(exc_value.args))
        #     except:
        #         pass
        # if traceback:
        #     try:
        #         pass#print("Warning: {}, {}".format(str(traceback.tb_frame), str(traceback.tb_lineno)))
        #     except:
        #         pass
        if self.__connect__:
            self.__connect__.close()
            self.__cursor__.close()    

    def getTickerPrice(self, hist_price_tb, symbol, mkt_name):
        sql_cmd = f'''select column_name from information_schema.columns where table_name= '{hist_price_tb}';'''
        self.__cursor__.execute(sql_cmd)
        cols = self.__cursor__.fetchall()

        sql_cmd = f"""
        SELECT * FROM {hist_price_tb} WHERE mkt_name in ({mkt_name}) AND symbol='{symbol}';
        """    
        # print(f'{sql_cmd}')
        self.__cursor__.execute(sql_cmd)
        x = self.__cursor__.fetchall()

        df = pd.DataFrame(data = x, columns = map(lambda x:x[0], cols))
        df['dateinfo'] = df['date']
        df.set_index('date', inplace= True)
        df.index = pd.to_datetime(df.index)
        df.replace('NaN', np.nan, inplace=True)
        df.fillna(method='bfill',inplace=True)
        for v in ['close', 'high', 'open', 'low', 'volume']:
            df[v] = pd.to_numeric(df[v])
        df.sort_index(inplace=True)

        return df

    def to_pd(self, hist_price_tb, output_pickle_file):
        sql_cmd = f'''select column_name from information_schema.columns where table_name= '{hist_price_tb}';'''
        self.__cursor__.execute(sql_cmd)
        cols = self.__cursor__.fetchall()

        sql_cmd = f"""
        SELECT * FROM {hist_price_tb} WHERE symbol in ('APPL','MSFT', 'TSLA', 'BABA');
        """    
        self.__cursor__.execute(sql_cmd)
        x = self.__cursor__.fetchall()

        df = pd.DataFrame(data = x, columns = map(lambda x:x[0], cols))
        df['dateinfo'] = df['date']
        df.set_index('date', inplace= True)
        df.index = pd.to_datetime(df.index)
        for v in ['close', 'high', 'open', 'low', 'volume']:
            df[v] = pd.to_numeric(df[v])
        df.sort_index(inplace=True)
        
        with open(output_pickle_file, 'wb') as fo:
            pickle.dump(df, fo)




if __name__ == '__main__':
    dbuser, dbname = 'gs', 'stock'
    stock_price_tb = 'stock_hist_price'

    with dbManager(dbuser, dbname) as dbm:
        symbol = '603222.SS'
        mkt_name = 'sse'
        # dbm.getTickerPrice(stock_price_tb, symbol, mkt_name)
        dbm.to_pd(stock_price_tb, '/home/gs/Work/fintek/stock_price_df.pkl')
