import logging
from pathlib import Path

def setLog(tick_symbol):

    logging_strategy_file = Path(__file__).parent/f'datas/{tick_symbol}_strategy_running_evaluation.log'
    # print(root_pth)
    # logging_strategy_file = f'/home/gs/Work/fintek/aimodel/datas/{tick_symbol}_strategy_running_evaluation.log'
    logging.basicConfig(filename=logging_strategy_file, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    # logging.info(infome(__file__, 'll'))

def infome(filename, msg):
    return f'{filename} - {msg}'


if __name__ == '__main__':
    setLog('TQQQ')