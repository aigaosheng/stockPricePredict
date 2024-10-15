import os
import httpx
import json
import subprocess
from abc import ABC
from typing import Callable, Union, Dict, Any, Union
from langchain_community.chat_models import ChatOllama
# from langchain_ollama.llms import OllamaLLM

class LongerThanContextError(Exception):
    pass

class ChatOpenAICompatible(ABC):
    def __init__(
        self,
        end_point: str,
        model = "llama3.1", #"phi3v", #
        system_message: str = "You are a helpful assistant.",
        other_parameters: Union[Dict[str, Any], None] = None,
    ):
        #[NOTICE]: DEV based ON ollama

        api_key = os.environ.get("OPENAI_API_KEY", "-")
        self.end_point = end_point
        self.model = model
        self.system_message = system_message
                
        if self.model.startswith("phi3") or self.model.startswith("llama"): #OLLAMA
            self.headers = {
                'Content-Type': 'application/json'
            }                  
        else:
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        self.other_parameters = {} if other_parameters is None else other_parameters

    def parse_response(self, response: httpx.Response) -> str:
        if self.model.startswith("gpt"):
            response_out = response.json()
            return response_out["choices"][0]["message"]["content"]
        elif self.model.startswith("phi3"):
            response_out = response.json()#[0]
            return response_out["response"]
        elif self.model.startswith("llama"):
            response_out = response.json()#[0]
            return response_out["response"]
        else:
            raise NotImplementedError(f"Model {self.model} not implemented")

    def guardrail_endpoint(self) -> Callable:
        def end_point(input: str, **kwargs) -> str:
            if kwargs.get("image", None):
                input_str = [
                    {
                        "role": "user", 
                        "content": [
                        {
                            "type": "text", 
                            "text": input
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": f'data:image/jpg;base64,{kwargs["image"]}'}
                        }
                        ]
                    }]
            else:
                input_str = [
                    {"role": "user", "content": input},
                ]
            
            if self.model.startswith("phi3"):
                payload = {
                    # "model": self.model,  
                    "prompt": "\n".join(map(lambda x: x['content'], input_str)),
                    "max_context_length": 4096,
                    "max_length": 4096,
                    "images": None,
                }
                # payload.update(self.other_parameters)
                payload = json.dumps(payload)                        
                response = httpx.post( #"http://0.0.0.0:6666/api/generate"
                    self.end_point, headers = self.headers, data = payload, timeout = 600.0
                )
            elif self.model.startswith("llama"):
                try:
                    llm = ChatOllama(base_url = self.end_point, 
                                     model = self.model, 
                                     temperature = self.other_parameters.get("temperature", 0.1),
                                     seed = self.other_parameters.get("seed", None),
                    )
                    resp = llm.invoke(input_str)
                    return resp.content
                    
                except:
                    return ''                    
            elif self.model.startswith("hugging-quants/Meta-Llama-3.1"):
                from openai import OpenAI
                # server_host = "http://100.92.49.113:8080"
                # model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
                api_key = 'TSGS@LLM'            
                client = OpenAI(
                    base_url = self.end_point + '/v1',
                    api_key = api_key,
                )
                response = client.chat.completions.create(
                        model=self.model,
                        messages = input_str
                    )
                return response.choices[0].message.content
            elif self.model.startswith("gpt-"):
                from openai import OpenAI
                client = OpenAI()
                response = client.chat.completions.create(
                        model = self.model,
                        messages = input_str,
                        max_tokens = self.other_parameters.get("max_tokens", 512),
                        temperature = self.other_parameters.get("temperature", 0.1),
                        seed = self.other_parameters.get("seed", None),
                )
                return response.choices[0].message.content
            
            else:
                payload = {
                    "model": self.model,  # or another model like "gpt-4.0-turbo"
                    "messages": input_str,
                }
                payload.update(self.other_parameters)
                payload = json.dumps(payload)
                        
                response = httpx.post(
                    self.end_point, headers = self.headers, data = payload, timeout = 600.0
                )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if (response.status_code == 422) and ("must have less than" in response.text):
                    raise LongerThanContextError
                else:
                    raise e

            return self.parse_response(response)

        return end_point

    def __call__(self, input: str, **kwargs) -> str:
        input_str = [
            # {"role": "system", "content": f"{self.system_message}"},
            # {"role": "system", "content": "You are a helpful assistant only capable of communicating with valid JSON, and no other text."},
            # {"role": "user", "content": f"{input}"},
            {"role": "user", "content": input,}
        ]
        
        if self.model.startswith("phi3"):
            payload = {
                # "model": self.model,  
                "prompt": "\n".join(map(lambda x: x['content'], input_str)),
                "max_context_length": 4096,
                "max_length": 4096,
                "images": None,
            }
            # payload.update(self.other_parameters)
            payload = json.dumps(payload)                        
            response = httpx.post( #"http://0.0.0.0:6666/api/generate"
                self.end_point, headers = self.headers, data = payload, timeout = 600.0
            )
        elif self.model.startswith("llama"):
            try:
                llm = ChatOllama(base_url = self.end_point, model = self.model, temperature = 0.1)
                resp = llm.invoke(input_str)
                return resp.content
                
            except:
                return ''                    
        elif self.model.startswith("hugging-quants/Meta-Llama-3.1"):
            from openai import OpenAI
            # server_host = "http://100.92.49.113:8080"
            # model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
            api_key = 'TSGS@LLM'            
            client = OpenAI(
                base_url = self.end_point + '/v1',
                api_key = api_key,
            )
            response = client.chat.completions.create(
                    model=self.model,
                    messages = input_str
                )
            return response.choices[0].message.content
        elif self.model.startswith("gpt-"):
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                    model = self.model,
                    messages = input_str,
                    max_tokens = 512,
                    temperature = 0.1,
            )
            return response.choices[0].message.content
        else:
            payload = {
                "model": self.model,  # or another model like "gpt-4.0-turbo"
                "messages": input_str,
            }
            payload.update(self.other_parameters)
            payload = json.dumps(payload)
                    
            response = httpx.post(
                self.end_point, headers = self.headers, data = payload, timeout = 600.0
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if (response.status_code == 422) and ("must have less than" in response.text):
                raise LongerThanContextError
            else:
                raise e

        return self.parse_response(response)
    
if __name__ == "__main__":
    server_host = 'http://100.92.49.113:8000'
    model_name ="llama3.1:70b-instruct-q4_K_M"

    server_host = "http://100.92.49.113:8080"
    model_name = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

    model_name = "gpt-4o-mini"
    llm = ChatOpenAICompatible(end_point=server_host, model = model_name)
    d = llm("tell me a joke")
    print(d)