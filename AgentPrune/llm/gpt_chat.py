import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from AgentPrune.llm.format import Message
from AgentPrune.llm.price import cost_count
from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry


load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    headers = {
        'Content-Type': 'application/json',
        'authorization': authorization_key
    }
    data = {
        "name": model,
        "inputs": {
            "stream": False,
            "msg": repr(msg),
        }
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers ,json=data) as response:
            response_data = await response.json()
            if isinstance(response_data['data'],str):
                prompt = "".join([item['content'] for item in msg])
                cost_count(prompt,response_data['data'],model)
                return response_data['data']
            else:
                raise Exception("api error")
    

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass