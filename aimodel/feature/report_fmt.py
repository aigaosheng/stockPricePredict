from instructor import OpenAISchema
from pydantic import BaseModel, create_model

class chart_report_fmt(OpenAISchema):
    """LLM output result"""
    reason: str
    signal: str
    entry_price: float
    exit_price: float

class chart_report_fmt_tsfa(OpenAISchema):
    """LLM output result"""
    reason_technical_analysis: str
    reason_news_sentiment: str
    reason_fundamental: str
    reason_analyst_recommend: str
    signal: str
    entry_price: float
    exit_price: float

class chart_report_fmt_t(OpenAISchema):
    """LLM output result"""
    reason_technical_analysis: str
    signal: str
    entry_price: float
    exit_price: float

class chart_report_fmt_tfa(OpenAISchema):
    """LLM output result"""
    reason_technical_analysis: str
    reason_fundamental: str
    reason_analyst_recommend: str
    signal: str
    entry_price: float
    exit_price: float

class chart_report_fmt_tsfa(OpenAISchema):
    """LLM output result"""
    reason_technical_analysis: str
    reason_news_sentiment: str
    reason_fundamental: str
    reason_analyst_recommend: str
    signal: str
    entry_price: float
    exit_price: float

class chart_report_fmt_ta(OpenAISchema):
    """LLM output result"""
    reason_technical_analysis: str
    reason_analyst_recommend: str
    signal: str
    entry_price: float
    exit_price: float

get_fmt_tool = {
    "ALL_IN": chart_report_fmt.openai_schema,
    "T": chart_report_fmt_t.openai_schema,
    "TSFA": chart_report_fmt_tsfa.openai_schema,
    "TFA": chart_report_fmt_tfa.openai_schema,
    "TA": chart_report_fmt_ta.openai_schema,
}

    