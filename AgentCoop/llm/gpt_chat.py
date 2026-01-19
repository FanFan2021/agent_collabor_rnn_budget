import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os
import pdb

from AgentCoop.llm.format import Message
from AgentCoop.llm.price import cost_count
from AgentCoop.llm.llm import LLM
from AgentCoop.llm.llm_registry import LLMRegistry

import random
from openai import AsyncOpenAI
import async_timeout
import asyncio


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



@retry(wait=wait_random_exponential(min=1, max=100), stop=stop_after_attempt(3))  # FIX: Faster backoff, more retries
async def achat(model: str, msg: List[Dict],):
    # api_key = random.sample(OPENAI_API_KEYS, 1)[0]
    # api_kwargs = dict(api_key = api_key)
    # pdb.set_trace()
    
    # aclient = AsyncOpenAI(**api_kwargs)
    aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)
    # pdb.set_trace()
    try:
        async with async_timeout.timeout(90):  # FIX: Reduced timeout from 120s to 90s for faster failure detection
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content
        # pdb.set_trace()
        if response_message is None:
            raise RuntimeError("API returned None content - model may have refused or returned empty response")
        
        prompt = "".join([item['content'] for item in msg])
        cost_count(prompt, response_message, model)
        
        return response_message

    except asyncio.TimeoutError as e:
        raise RuntimeError(f"API call timed out after 90s: {e}")  # FIX: Better error message
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()  # FIX: Get full traceback
        raise RuntimeError(f"Failed to complete the async chat request: {e}\n{error_details}")


# OPENAI_API_KEYS = ['']
# BASE_URL = ''

# load_dotenv()
# MINE_BASE_URL = os.getenv('BASE_URL')
# MINE_API_KEYS = os.getenv('API_KEY')


# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
# async def achat(
#     model: str,
#     msg: List[Dict],):
#     request_url = MINE_BASE_URL
#     authorization_key = MINE_API_KEYS
#     headers = {
#         'Content-Type': 'application/json',
#         'authorization': authorization_key
#     }
#     data = {
#         "name": model,
#         "inputs": {
#             "stream": False,
#             "msg": repr(msg),
#         }
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(request_url, headers=headers ,json=data) as response:
#             response_data = await response.json()
#             prompt = "".join([item['content'] for item in msg])
#             cost_count(prompt,response_data['data'],model)
#             return response_data['data']

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
        
        # pdb.set_trace()
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