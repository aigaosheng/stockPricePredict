import pandas as pd
import toml
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import pytz

sys.path.append(str(Path(os.path.abspath(__file__)).parent))
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))

# from priceLoader.finnhub_load import price_fetch
# from priceLoader.polygon_load import price_fetch
from priceLoader.finnhub_db_load import price_fetch

from llmsignal_v2.agent import Agent
from utils_shared.db_client import SQLClient
from tqdm import tqdm
import logging

logging.basicConfig(format = '%(asctime)s - %(name)s: %(filename)s - Line:%(lineno)d, %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def batch_extract_signal(config_file):
    cfg = toml.load(config_file)
    price_df = price_fetch(cfg["backtest"]["symbol"][0], dt_from = datetime.fromisoformat(cfg["backtest"]["dt_from"]),dt_to = datetime.fromisoformat(cfg["backtest"]["dt_to"]))
    dt_seq = price_df.index.map(lambda x: pd.to_datetime(x))

    tb_name = "ta_signal_llm_fusion"
    # client = SQLClient('postgres', 'gs', "localhost", "5432", "stock")
    client = SQLClient('postgres', 'postgres', "localhost", "5432", "postgres")

    #Set LLM model here
    # args = cfg["chatgpt"] #["ollama"]
    args = cfg["ollama"]

    inst = Agent(**args)
    # input_src = "ta" #"ta+news"
    # source_price = "finnhub"
    input_src = "ta"
    if cfg["backtest"]["args"]["use_fundamental"]:
        input_src += "_fund"
    
    if cfg["backtest"]["args"]["use_news"]:
        input_src += "_senti"

    if cfg["backtest"]["args"]["use_analyst"]:
        input_src += "_analyst"

    for symbol in cfg["backtest"]["symbol"]:
        for dt_now in tqdm(dt_seq):
            resp = inst(symbol, dt_to = dt_now.strftime("%Y-%m-%d"), n_lookback_day = 5, **cfg["backtest"]["args"])#"yfinance")

            # if input_src == "ta+news":
            #     resp = inst(symbol, dt_to = dt_now.strftime("%Y-%m-%d"), n_lookback_day = 5, use_fundamental = True, use_news = True, use_analyst = True, source_price = source_price)#"yfinance")
            # else:
            #     resp = inst(symbol, dt_to = dt_now.strftime("%Y-%m-%d"), n_lookback_day = 5, use_fundamental = False, use_news = False, use_analyst = False, source_price = source_price)#"yfinance")
            # print(json.dumps(resp, indent = 2))

            #
            resp_stack = [resp]
            resp_stack = pd.DataFrame(resp_stack)
            cols = resp_stack.columns.to_list()
            resp_stack['symbol'] = symbol
            cols_ord = ["symbol"] + cols
            resp_stack["dt"] = pytz.timezone("UTC").localize(dt_now) 
            cols_ord += ["dt"]
            resp_stack["created_at"] = pytz.timezone("UTC").localize(datetime.now())
            cols_ord += ["created_at"]
            resp_stack["model_name"] = args["model_name"]
            cols_ord += ["model_name"]

            resp_stack["input_src"] = input_src
            cols_ord += ["input_src"]

            resp_stack = resp_stack[cols_ord]
            #
            if (tb_name is not None) and (client is not None):
                try:
                    client.create_table(resp_stack, tb_name)
                except Exception as e:
                    print(f"Warning: create table -> {tb_name} -> {e}")

                try:
                    client.update_db(resp_stack, tb_name)
                except Exception as e:
                    print(f"Warning: write table -> {tb_name} -> {e}")
            logger.info(f"Processing {symbol} on {dt_now}")
            # break

if __name__ == "__main__":
    config_file = str(Path(os.path.abspath(__file__)).parent/"batch_config.toml")
    batch_extract_signal(config_file)