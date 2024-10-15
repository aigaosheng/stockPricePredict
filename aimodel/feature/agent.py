from dotenv import load_dotenv
load_dotenv(override=True)

import json
import guardrails as gd
from pydantic import BaseModel, Field
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent.parent))
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent.parent.parent))

from pydantic import BaseModel, Field
from typing import List
from llmsignal_v2.price2ti import prepare_prompt
import logging
import time
import toml
import copy
from datetime import datetime
from chat import ChatOpenAICompatible

logging.basicConfig(format = '%(asctime)s - %(name)s: %(filename)s - Line:%(lineno)d, %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalResponseGr_V1(BaseModel):
    reason_technical_analysis: str = Field(description = "step-by-step in-depth technical analysis and detailed reasoning")
    reason_news_sentiment: str = Field(description = "step-by-step analysis and reasoning of sentiment's impact on price movement")
    reason_fundamental: str = Field(description = "step-by-step analysis and reasoning of company fundamental data's impact on price movement")
    reason_analyst_recommend: str = Field(description = "step-by-step analysis and reasoning of analyst recommendation's impact on price movement")
    signal: str = Field(description = "Recommendation action, a value in strong buy, buy, hold, sell, and strong sell")
    entry_price: float = Field(description = "entry price")
    exit_price: float = Field(description = "exit price")

class TechnicalResponseGr(BaseModel):
    reason_technical_analysis: str = Field(description = "step-by-step in-depth technical analysis and detailed reasoning")
    reason_news_sentiment: str = Field(description = "step-by-step analysis and reasoning of sentiment's impact on price movement")
    reason_fundamental: str = Field(description = "step-by-step analysis and reasoning of company fundamental data's impact on price movement")
    reason_analyst_recommend: str = Field(description = "step-by-step analysis and reasoning of analyst recommendation's impact on price movement")
    signal: str = Field(description = "one value among strong buy, buy, hold, sell, and strong sell")
    entry_price: float = Field(description = "entry price")
    exit_price: float = Field(description = "exit price")

class Agent:
    def __init__(self, model_name = "llama3.1:8b-instruct-q5_K_M", end_point = "http://localhost:11434"):
        self.model_name = model_name
        self.end_point = end_point
        self.system_message = """You act as an expert trader. You leverage various technical analysis methods such as oscillators, moving average, chart pattern, and pivot points. You also collect information such the company fundamental data, news sentiment data, analyst recommendation, and investigate how they impact on the price movement. You analyze, reason, and forecast price movement ONLY based on the given information. 

        """

        self.output_fmt_gc = """ 
        Output the results using the following schema. If INFORMATION NOT AVAILABLE IN THE FIELD, OUTPUT `NULL`, DO NOT MAKE UP.

        ${gr.complete_xml_suffix_v2}
        Your output should strictly conforms the following xml format without any additional contents: {"reason_technical_analysis": string, "reason_news_sentiment": string, "reason_fundamental": string, "reason_analyst_recommend": string, "signal": string, "entry_price": number, "entry_price": number}
        """
        ta_config_file = str(Path(os.path.abspath(__file__)).parent/"config_ta.toml")
        self.ta_cfg = toml.load(ta_config_file)
        logger.info(f"Load technical indicator configure from {ta_config_file}")

        if model_name.startswith("gpt-"):
            self.guardrail_endpoint = ChatOpenAICompatible(
                end_point = "",
                model = model_name,
                # system_message = "",
                other_parameters = {
                    "temperature": 0.1,
                    "max_tokens": 768,
                },
            ).guardrail_endpoint()
        else:
            self.guardrail_endpoint = ChatOpenAICompatible(
                end_point = end_point,
                model = model_name,
                # system_message = "",
                other_parameters = {"temperature":0.1},
            ).guardrail_endpoint()


    def __call__(self, symbol, dt_to: str = "2024-08-24", use_fundamental = True, use_news = True, use_analyst = True, use_chartist_text = True, use_chartist_image = True, source_price = "finnhub"):
        # import litellm
        from guardrails import Guard

        ta_cfg_now= copy.deepcopy(self.ta_cfg)
        logger.info(f"** Price source -> {source_price}")

        dt_to = dt_to if dt_to else datetime.now().strftime("%Y-%m-%d")

        # prompt_msg, price_df, (use_fundamental, use_news, use_analyst) = prepare_prompt(symbol, dt_to = dt_to, window_len = 365, use_fundamental = use_fundamental, use_news = use_news, use_analyst = use_analyst, ta_config = ta_cfg_now, source_price = source_price)
        prompt_context = prepare_prompt(symbol, dt_to = dt_to, window_len = 365, use_fundamental = use_fundamental, use_news = use_news, use_analyst = use_analyst, ta_config = ta_cfg_now, use_chartist_text = use_chartist_text, use_chartist_image = use_chartist_image, source_price = source_price)

        prompt_msg = prompt_context["user_prompt"]
        price_df = prompt_context["price_df"]
        use_fundamental = prompt_context["use_fundamental"]
        use_news = prompt_context["use_news"]
        use_analyst = prompt_context["use_analyst"]
        chartist_image = prompt_context["image"] if use_chartist_image else None
        
        # price_df_used = price_df.tail(n_lookback_day)[["high", "open", "low", "close"]]
        if price_df is None:
            return {
                "reason_technical_analysis": None,
                "reason_fundamental": None,
                "reason_news_sentiment": None,
                "reason_analyst_recommend": None,
                "signal": None, 
                "entry_price": None, 
                "entry_price": None,
                "status": "invalid", 
                "raw": "",
                "is_multimodal": False
            }
        today = price_df.index[-1].strftime("%Y-%m-%d")

        price_df_prompt = "" # f"You are given the {n_lookback_day}-day historical price of {symbol}, in DATAFRAME FORMAT, with columns: date, high, open, low, close.\n{price_df_prompt} ** Historical price\n{price_df_used}\n"

        investment_info = f"""You are currently trading the stock {symbol}. 
        {price_df_prompt} 

        You collect summarized data from 1) technical analysis, 2) fundamental financial metric, 3) news sentiment, and 4) analyst recommendation, which are shown in the following.

        {prompt_msg}

        Today is {today}. Based on these information and your expert trading knowledge, you do thorough analysis on the stock, {symbol}, reasoning the underlying factor to drive price movement, recommend trading action (i.e. trading action: strong buy, buy, hold, sell and strong sell), suggest entry price and exit price.   

        """
        
        prompt_now = self.system_message + "\n" + investment_info + "\n" + self.output_fmt_gc

        logger.info(f"** Prompt ready -> {prompt_now[:30]}")

        guard = Guard.from_pydantic(
            output_class = TechnicalResponseGr, 
            prompt = prompt_now
        )
         
        # Call the Guard to wrap the LLM API call
        if self.model_name.startswith("gpt"):
            import openai
            try:
                st0 = time.time()
                # validated_response = guard(
                #     openai.chat.completions.create,
                #     model = self.model_name,
                #     max_tokens = 1024,
                #     # msg_history = [{"role": "user", "content": prompt_now}],      

                #     msg_history = msg_history,
              
                #     temperature = 0.1, #0.0,
                #     num_reasks = 1                
                # )

                validated_response = guard(
                    # litellm.completion,
                    self.guardrail_endpoint,
                    output_class = TechnicalResponseGr, 
                    prompt = prompt_now, 
                    image = chartist_image,
                    reask = 1
                )

                logger.info(f"** GPT inference time -> {round(time.time() - st0, 4)}s")
                if validated_response.validation_passed:
                    output = validated_response.validated_output
                    output["raw"] = ""
                    output["status"] = "success"
                    logger.info(f"** GPT inference JSON OK")
                else:
                    try:
                        output = json.loads(validated_response.raw_llm_output)
                        output["raw"] = ""
                        output["status"] = "success"
                        logger.info(f"** GPT inference JSON OK from raw")
                    except:
                        output = {"raw": validated_response.raw_llm_output}
                        output["status"] = "fail"
                        logger.info(f"** GPT inference failure. Output Non-JSON raw content")

            except Exception as e:
                output = {"status": "fail", "raw": ""}
                logger.info(f"** GPT inference failure. NO response. {e}")
        else: #ollama
            try:
                st0 = time.time()
                # validated_response = guard(
                #     litellm.completion,
                #     model = f"ollama/{self.model_name}", #"ollama/phi3.5:3.8b-mini-instruct-fp16", #
                #     max_tokens = 512,
                #     api_base = self.end_point, #"http://localhost:11434",
                #     msg_history = [
                #         {"role": "user", "content": prompt_now}
                #     ],                    
                #     temperature = 0.1,
                #     num_reasks = 1                
                # )

                validated_response = guard(
                    # litellm.completion,
                    self.guardrail_endpoint,
                    output_class = TechnicalResponseGr, 
                    prompt = prompt_now, 
                    reask = 1
                )

                logger.info(f"** Ollama inference time -> {round(time.time() - st0, 4)}s")

                if validated_response.validation_passed:
                    output = validated_response.validated_output
                    output["raw"] = ""
                    output["status"] = "success"
                    logger.info(f"** Ollama inference JSON OK")
                else:
                    try:
                        output = json.loads(validated_response.raw_llm_output)
                        output["raw"] = ""
                        output["status"] = "success"
                        logger.info(f"** Ollama inference JSON OK from raw")
                    except:
                        output = {"raw": validated_response.raw_llm_output}
                        output["status"] = "fail"
                        logger.info(f"** Ollama inference failure in JSON. Outout raw NON-JSON content")

            except:
                output = {"status": "fail", "raw": ""}
                logger.info(f"** Ollama inference failure. NO Response")

        if not use_fundamental:
            output["reason_fundamental"] = None
        if not use_news:
            output["reason_news_sentiment"] = None
        if not use_analyst:
            output["reason_analyst_recommend"] = None

        if chartist_image:
            output["is_multimodal"] = True
        else:
            output["is_multimodal"] = False

        return output    

if __name__ == "__main__":
    symbol = "AZO" #"NVDA" #
    inst = Agent(model_name = "gpt-4o-mini")
    # inst = Agent(model_name = "llama3.1:8b-instruct-q5_K_M", end_point="http://localhost:21434")
    st1 = time.time()
    x = inst(symbol, dt_to="2024-10-14",  use_fundamental = True, use_news = True, use_analyst = True, use_chartist_text = False, use_chartist_image = True)

    print(f"** Spending time -> {time.time() - st1}")
    print(json.dumps(x, indent = 2))
