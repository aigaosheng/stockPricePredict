#*-* coding: utf-8 *-*
#!/usr/bin/env python3

'''
long-short strategy evaluation
'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtrader as bt
import backtrader.indicators as btind
from backtrader.feeds import GenericCSVData

import pandas as pd
import quantstats
from backtrader.analyzers import (SQN, AnnualReturn, TimeReturn, SharpeRatio,TradeAnalyzer)

import logging
from .melog import infome
from .utils_core import figure2base64, image2html, html2image


# logging_strategy_file = '/home/gs/Work/fintek/aimodel/datas/{symbol}_strategy_running_evaluation.log'


class MLSignal(bt.Indicator):

    lines = ('predict',)

    def __init__(self):
        self.lines.predict = self.data0.predict

class MLPredictCsv(bt.feeds.PandasData):
    '''
    Desc: for customized CSV format
    '''

    # What new data will be availible in the Stratagies line object
    lines = ('predict',)

    # Which columns go to which variable
    params = (
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('Adj Close','Adj Close'),
        ('volume', 'volume'),
        ('openinterest', 'openinterest'),
        ('predict', 'predict'),
    )

# Create a Stratey
class MLStrategy(bt.Strategy):
    params = dict(
        onlylong = False
    )
    
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.ml_signal = MLSignal(self.data)
        # self.buysig = self.datas[0].predict 
        # To keep track of pending orders
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                # self.log('BUY EXECUTED, %.2f' % order.executed.price)
                logging.info(infome(__file__,'BUY EXECUTED, %.2f' % order.executed.price))
            elif order.issell():
                # self.log('SELL EXECUTED, %.2f' % order.executed.price)
                logging.info(infome(__file__,'SELL EXECUTED, %.2f' % order.executed.price))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # self.log('Order Canceled/Margin/Rejected')
            logging.info(infome(__file__,'Order Canceled/Margin/Rejected'))

        # Write down: no pending order
        self.order = None

    def next(self):
        #if order is active, no new order allow
        if self.order:
            return

        # Check if we are in the market
        if self.ml_signal.lines.predict > 0:
            if self.position:
                # self.log('CLOSE SHORT , %.2f' % self.data.close[0])
                logging.info(infome(__file__,'CLOSE SHORT , %.2f' % self.data.close[0]))
                self.close()
            # Buy
            # self.log('BUY CREATE, %.2f' % self.dataclose[0])
            logging.info(infome(__file__,'BUY CREATE, %.2f' % self.dataclose[0]))
            self.order = self.buy()
        elif self.ml_signal.lines.predict < 0:
            if self.position:
                # self.log('CLOSE LONG , %.2f' % self.data.close[0])
                logging.info(infome(__file__,'CLOSE LONG , %.2f' % self.data.close[0]))
                self.close()
            
            if not self.p.onlylong:
                # self.log('SELL CREATE , %.2f' % self.data.close[0])
                logging.info(infome(__file__,'SELL CREATE , %.2f' % self.data.close[0]))
                self.sell()

def strategyEvaluate(tick_symbol, ml_predict_csv, strategy_log_file = None, quant_output = None, quant_output_html = None, n_stake = 40, cash_capital = 1000, is_onlylong = False, is_model_opt = False):
    '''
    Desc: ML predict signal based strategy evaluation

    return: money reward. If is_model_opt = true, do not show performance report

    '''

    mlpredicted_signal = pd.read_csv(ml_predict_csv,
                                parse_dates=True,
                                index_col=0,                             
                            )
    def mapsignal_multi(x):
        if x == 0:
            return 0 #bt.SIGNAL_NONE
        elif x == 1:
            return 1 #bt.SIGNAL_LONG
        elif x == 2:
            return -1 #bt.SIGNAL_SHORT

    def mapsignal_binary(x):
        if x == 0:
            return -1 #bt.SIGNAL_NONE
        elif x == 1:
            return 1 #bt.SIGNAL_LONG

    # print(f'*** {ml_predict_csv}, {mlpredicted_signal.columns}')
    if max(mlpredicted_signal['predict'])>1:
        mapsignal = mapsignal_multi
    else:
        mapsignal = mapsignal_binary

    mlpredicted_signal['predict'] = mlpredicted_signal['predict'].transform(mapsignal)
    # print(f'*** {mlpredicted_signal.columns}')

    yr1, mth1, day1 = list(map(lambda x:int(x),str(mlpredicted_signal.index[0]).split(' ')[0].split('-')))
    yr2, mth2, day2 = list(map(lambda x:int(x),str(mlpredicted_signal.index[-1]).split(' ')[0].split('-')))
    
    mlpredicted_signal['openinterest'] = 0
    data = MLPredictCsv(dataname=mlpredicted_signal)

    # create Cerebro instance and attach data to it
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    # Add a strategy
    cerebro.addstrategy(MLStrategy, onlylong = is_onlylong)
    # Set our desired cash start
    cerebro.broker.setcash(cash_capital)

#     cerebro.broker.setcommission(commission=0)

    #Add strategy to Cerebro
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
    cerebro.addanalyzer(TradeAnalyzer)

    # better net liquidation value view
    cerebro.addobserver(bt.observers.Value)

    # Default position size
    cerebro.addsizer(bt.sizers.SizerFix, stake=n_stake)    
    
    #add output log file 
    if not is_model_opt:
        cerebro.addwriter(bt.WriterFile, csv=True, out=strategy_log_file)    


    # Print out the starting conditions
    capital_start = cerebro.broker.getvalue()
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    results = cerebro.run()

    # Print out the final result
    capital_end = cerebro.broker.getvalue()
    money_reward = capital_end - capital_start
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    #Get strategy stats
    strat = results[0]
    portfolio_stats = strat.analyzers.getbyname('PyFolio')
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)
    sharpe_ratio = quantstats.stats.sharpe(returns)
    
    # print(returns)
    # print(positions)
    # print(portfolio_stats)
    if not is_model_opt:
        quantstats.reports.html(returns, output = quant_output, download_filename = quant_output_html, title = tick_symbol)
        
        figs = cerebro.plot(iplot=False)
        img_lst = []
        for fg in figs:
            fg_img = []
            for fga in fg:
                fg_img.append(figure2base64(fga))
            img_lst.append(fg_img)

        quant_img = html2image(quant_output_html)
        img_lst.append([quant_img])
        img_html = image2html(img_lst)
        #image to html
        with open(quant_output_html, 'wt') as fi:
            fi.write(img_html)

        import webbrowser
        webbrowser.open(quant_output_html)

        o_pth = os.path.dirname(quant_output_html)
        o_fl_name = quant_output_html.split('/')[-1].split('.')[0]
        o_merge_file = os.path.join(o_pth, o_fl_name + '.jpg')

        quant_img = html2image(quant_output_html, o_merge_file)
        # webbrowser.open(o_merge_file)
        
    return money_reward #sharpe_ratio #

if __name__ == '__main__':
    tfgaick_symbol = 'TQQQ'
    signal_file = f'/home/gs/Work/fintek/aimodel/datas/{tick_symbol}_predict_eval_test.csv'
    o_strategy_log_file = f'datas/{tick_symbol}_predict_eval_strategy.log'
    o_quant_output_file = f'datas/{tick_symbol}_predict_eval_strategy.stats'
    o_quant_output_html = f'datas/{tick_symbol}_predict_eval_strategy.html'
    n_stake = 10
    cash_capital = 1000
    is_onlylong = False #True

    strategyEvaluate(tick_symbol, signal_file, o_strategy_log_file, o_quant_output_file, o_quant_output_html, n_stake, cash_capital, is_onlylong)