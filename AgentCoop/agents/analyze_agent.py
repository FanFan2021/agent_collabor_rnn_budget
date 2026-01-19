from typing import List,Any,Dict
import re
import pdb
from AgentCoop.graph.node import Node
from AgentCoop.agents.agent_registry import AgentRegistry
from AgentCoop.llm.llm_registry import LLMRegistry
from AgentCoop.prompt.prompt_set_registry import PromptSetRegistry
from AgentCoop.tools.search.wiki import search_wiki_main
from AgentCoop.graph.constants import ADV_ROLES
import torch
import random

def find_strings_between_pluses(text):
    return re.findall(r'\@(.*?)\@', text)

@AgentRegistry.register('AnalyzeAgent')
class AnalyzeAgent(Node):
    def __init__(self, id: str | None =None, role:str = None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "AnalyzeAgent" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        if role is not None:
            self.original_role = role
        else:
            self.original_role = self.prompt_set.get_role()
        self.role = self.original_role
        self.wiki_summary = ""
        
    async def _process_inputs(self, raw_inputs:Dict[str,str], comm_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """ 
        
        constraint = self.prompt_set.get_analyze_constraint(self.role)

        system_prompt = f"{constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n"
        comm_str = ""
        for id, info in comm_info.items():
            comm_str += f"Agent {id}, role is {info['original_role']}, output is:\n\n {info['output']}\n\n"
            
        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{comm_str} \n\n" if len(comm_str) else ""
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  comm_info:Dict[str,Dict], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, comm_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  comm_info:Dict[str,Dict], adversarial_answer:str = None, **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        if self.role in ADV_ROLES:
            system_prompt = self.prompt_set.get_adversarial_system_prompt()
            user_prompt = self.prompt_set.get_adversarial_answer_prompt(input=input, answer=adversarial_answer)
            # pdb.set_trace()
        else:
            system_prompt, user_prompt = await self._process_inputs(input, comm_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        # pdb.set_trace()
        response = await self.llm.agen(message)
        if self.wiki_summary != "":
            response += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""
        # print(f"################system prompt:{system_prompt}")
        # print(f"################user prompt:{user_prompt}")
        # print(f"################response:{response}")
        return response

    # async def _async_execute(self, input:Dict[str,str], comm_info:Dict[str,Dict],**kwargs):
    #     num_comm_neighbors = len(comm_info)
    #     num_comm_adv = 0

    #     if self.role in ADV_ROLES:
    #         system_prompt = self.prompt_set.get_adversarial_system_prompt()
    #         user_prompt = self.prompt_set.get_adversarial_answer_prompt(input['task'])
    #     else:
    #         # 1. Count adversarial neighbors
    #         adv_comm_neighbors = []
    #         for info in comm_info.values():
    #             if info['role'] in ADV_ROLES:
    #                 num_comm_adv += 1
    #                 adv_comm_neighbors.append(info['output'])
            
    #         # 2. Calculate probabilities with safety checks
    #         comm_ratio = num_comm_adv / num_comm_neighbors if num_comm_neighbors > 0 else 0
            
    #         # 3. Stochastic Flip Logic (Comm)
    #         if comm_ratio > 0 and torch.rand(1).item() < comm_ratio:
    #             # pdb.set_trace()
    #             return random.choice(adv_comm_neighbors)
            
    #         # 4. Normal Path: Process inputs only if we didn't flip
    #         else:
    #             system_prompt, user_prompt = await self._process_inputs(input, comm_info)
        
    #     # Generate response via LLM
    #     message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
    #     response = await self.llm.agen(message)
        
    #     if self.wiki_summary != "":
    #         response += f"\n\n{self.wiki_summary}"
    #         self.wiki_summary = ""
            
    #     return response