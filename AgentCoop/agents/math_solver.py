from typing import List,Any,Dict

from AgentCoop.graph.node import Node
from AgentCoop.agents.agent_registry import AgentRegistry
from AgentCoop.llm.llm_registry import LLMRegistry
from AgentCoop.prompt.prompt_set_registry import PromptSetRegistry
from AgentCoop.tools.coding.python_executor import execute_code_get_return
from dataset.gsm8k_dataset import gsm_get_predict
from AgentCoop.graph.constants import ADV_ROLES
import re
import pdb
import os

@AgentRegistry.register('MathSolver')
class MathSolver(Node):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "",):
        super().__init__(id, "MathSolver" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        # pdb.set_trace()
        # self.role = self.prompt_set.get_role() if role is None else role
        if role is not None:
            self.original_role = role
        else:
            self.original_role = self.prompt_set.get_role()
        self.role = self.original_role
        # self.constraint = self.prompt_set.get_constraint(self.role) 
        
    def _process_inputs(self, raw_inputs:Dict[str,str], comm_info:Dict[str,Dict],**kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """             
        constraint = self.prompt_set.get_analyze_constraint(self.original_role)

        spatial_str = ""
        system_prompt = f"{constraint}"
       
        # user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["question"],role=self.original_role)
        user_prompt = f"The task is: {raw_inputs['question']}\n"
        # if self.role == "Math Solver":
        #     # Only add hint if there are actual hints from other agents
        #     hints = []
        #     for id, info in comm_info.items():
        #         if info.get("output"):
        #             hints.append(gsm_get_predict(info["output"]))
        #     if hints:
        #         user_prompt += "(Hint: The answer is near to " + " ".join(hints) + ")."
        # else:
        for id, info in comm_info.items():
            spatial_str += f"Agent {id} as a {info['original_role']} his answer to this question is:\n\n{info['output']}\n\n"
        
        user_prompt += f"At the same time, there are the following responses to the same question for your reference:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
             
        return system_prompt, user_prompt
    
    def _execute(self, input:Dict[str,str],  comm_info:Dict[str,Any], adversarial_answer: str = None, **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(input, comm_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response
    


    def write_to_file(self, response: str, filename: str):
        """
        Saves the agent's response to a specific file using a context manager.
        """
        # Optional: Ensure the directory exists
        os.makedirs("logs", exist_ok=True)
        filepath = os.path.join("logs", filename)

        with open(filepath, "a", encoding="utf-8") as f:
            # We use "a" (append) so that you keep a history of all rounds
            f.write(f"--- Response Entry ---\n{response}\n\n")

    async def _async_execute(self, input:Dict[str,str],  comm_info:Dict[str,Any], adversarial_answer: str = None, **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        if self.role in ADV_ROLES:
            system_prompt = self.prompt_set.get_adversarial_system_prompt()
            user_prompt = self.prompt_set.get_adversarial_answer_prompt(input, adversarial_answer)
            # pdb.set_trace()
        else:
            system_prompt, user_prompt = self._process_inputs(input, comm_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
                # Get the initial response (already a string based on your setup)
        response = await self.llm.agen(message)
        
        # Debug: check if response is None
        if response is None:
            pdb.set_trace()
            raise ValueError(f"LLM returned None for role={self.role}. Check API key, model name, or network connection.")

        # if self.role == "Math Solver":
        #     self.write_to_file(response, "math_solver_response.txt")

        max_retries = 2
        
        if self.role == "Programming Expert":
            for attempt in range(max_retries):
                try:
                    # Clean markdown backticks safely
                    clean_code = response.replace("```python", "").replace("```", "").strip()
                    
                    # Execute code logic
                    answer = execute_code_get_return(clean_code)
                    
                    if answer is not None and not str(answer).startswith("Error"):
                        response += f"\nthe answer is \\boxed{{{answer}}}"
                        return response
                    
                    # If we got an error message or None, trigger the retry logic
                    if attempt < max_retries - 1:
                        error_feedback = answer if answer else "No 'answer' variable was defined."
                        retry_message = message + [
                            {'role': 'assistant', 'content': response},
                            {'role': 'user', 'content': f"Execution failed. Error: {error_feedback}. Please fix the code logic."}
                        ]
                        response = await self.llm.agen(retry_message)
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        retry_message = message + [
                            {'role': 'assistant', 'content': response},
                            {'role': 'user', 'content': f"The code encountered a system error: {e}. Please rewrite it."}
                        ]
                        response = await self.llm.agen(retry_message)
            
            # 2. Final Fallback Logic (GSM8K specific)
            # Pull text from 'input' to avoid NameError: problem_text
            original_question = input.get('question', user_prompt)
            numbers = re.findall(r'\d+', original_question)
            fallback_val = max(set(numbers), key=numbers.count) if numbers else "0"
            
            response += f"\n[All retries failed]\nthe answer is \\boxed{{{fallback_val}}}"
            return response
            
        else:
            # Standard flow for text-based agents (Math Solver, Inspector, etc.)
            return response