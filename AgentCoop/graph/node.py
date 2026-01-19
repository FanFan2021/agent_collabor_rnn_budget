import shortuuid
from typing import List, Any, Optional,Dict
from abc import ABC, abstractmethod
import warnings
import asyncio
import torch
import pdb
from AgentCoop.graph.constants import ADV_ROLES, adv_add_on
import os

class Node(ABC):
    """
    Represents a processing unit within a graph-based framework.

    This class encapsulates the functionality for a node in a graph, managing
    connections to other nodes, handling inputs and outputs, and executing
    assigned operations. It supports both individual and aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the node.
        agent_type(str): Associated agent name for node-specific operations.
        spatial_predecessors (List[Node]): Nodes that precede this node in the graph.
        spatial_successors (List[Node]): Nodes that succeed this node in the graph.
        inputs (List[Any]): Inputs to be processed by the node.
        outputs (List[Any]): Results produced after node execution.
        raw_inputs (List[Any]): The original input contains the question or math problem.
        last_memory (Dict[str,List[Any]]): Input and output of the previous timestamp.
        
    Methods:
        add_predecessor(operation): 
            Adds a node as a predecessor of this node, establishing a directed connection.
        add_successor(operation): 
            Adds a node as a successor of this node, establishing a directed connection.
        memory_update():
            Update the last_memory.
        get_spatial_info():
            Get all of the info from spatial spatial_predecessors.
        execute(**kwargs): 
            Processes the inputs through the node's operation, handling each input individually.
        _execute(input, **kwargs): 
            An internal method that defines how a single input is processed by the node. This method should be implemented specifically for each node type.
        _process_inputs(raw_inputs, spatial_info, temporal_info, **kwargs)->List[Any]:
            An internal medthod to process the raw_input, the spatial info and temporal info to get the final inputs.
    """

    def __init__(self, 
                 id: Optional[str],
                 agent_name:str="",
                 domain:str="", 
                 llm_name:str = "",
                 ):
        """
        Initializes a new Node instance.
        """
        self.id:str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name:str = agent_name
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.predecessors: List[Node] = []
        self.successors: List[Node] = []
        self.predecessors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = ""
        self.original_role = ""
        self.last_memory: Dict[str,List[Any]] = {'outputs':[],'solution':[], 'analysis_embed':[],'reliable_scores':[], 'neighbors':[], 'rnn_hidden_states':[]}   
        self.first_memory: Dict[str,List[Any]] = {'outputs':[],'solution':[], 'analysis':[], 'solution_embed':[], 'analysis_embed':[]}      

    @property
    def node_name(self):
        return self.__class__.__name__
    
    def add_predecessor(self, operation: 'Node'):
        if operation not in self.predecessors:
            self.predecessors.append(operation)
            operation.successors.append(self)

    def add_successor(self, operation: 'Node'):
        if operation not in self.successors:
            self.successors.append(operation)
            operation.predecessors.append(self)

    def remove_predecessor(self, operation: 'Node'):
        if operation in self.predecessors:
            self.predecessors.remove(operation)
            operation.successors.remove(self)

    def remove_successor(self, operation: 'Node'):
        if operation in self.successors:
            self.successors.remove(operation)
            operation.predecessors.remove(self)

    def clear_connections(self):
        self.predecessors: List[Node] = []
        self.successors: List[Node] = []
    
    # def update_memory(self):
    #     self.last_memory['inputs'] = self.inputs
    #     self.last_memory['outputs'] = self.outputs
    #     self.last_memory['raw_inputs'] = self.raw_inputs
    #     self.last_memory['analysis'] = self.analysis
    #     self.last_memory['reliable_scores'] = self.reliable_scores


    def get_comm_info(self)->Dict[str,Dict]:
        """ Return a dict that maps id to info. """
        info = {}
        if self.predecessors is not None:
            for predecessor in self.predecessors:
                predecessor_outputs = predecessor.outputs
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                # Skip predecessors with None output
                if predecessor_output is None:
                    continue
                info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output.strip(), "original_role":predecessor.original_role}

        return info
    
    def execute(self, input:Any, adversarial_answers:Dict[str,str]={}, **kwargs):
        self.outputs = []
        comm_info:Dict[str,Dict] = self.get_comm_info()
        
        # Check if this is an adversarial node with a fixed answer
        if getattr(self, "role", None) in ADV_ROLES and self.id in adversarial_answers:
            # Get the fixed adversarial answer
            fixed_answer = adversarial_answers[self.id]
            
            # Still generate analysis to make it look realistic
            results = [self._execute(input, comm_info, **kwargs)]
            
            # Replace the answer part with the fixed adversarial answer
            for result in results:
                if not isinstance(result, list):
                    result = [result]
                for res in result:
                    # Extract analysis part (everything after first character which is the answer)
                    if len(res) > 1:
                        analysis = res.strip()
                        # Combine fixed answer with analysis
                        modified_result = f"{fixed_answer}\n\n{analysis}"
                    else:
                        # Just use the fixed answer if no analysis
                        # modified_result = fixed_answer
                        pdb.set_trace()
                        raise ValueError("No analysis found")
                    self.outputs.append(modified_result)
        else:
            # Normal execution for non-adversarial agents
            results = [self._execute(input, comm_info, **kwargs)]

            for result in results:
                if not isinstance(result, list):
                    result = [result]
                self.outputs.extend(result)
        return self.outputs


    async def async_execute(self, input:Any, adversarial_answers:Dict[str,str]={}, **kwargs):

        self.outputs = []
        comm_info:Dict[str,Dict] = self.get_comm_info()
        
        # Check if this is an adversarial node with a fixed answer
        if getattr(self, "role", None) in ADV_ROLES and self.id in adversarial_answers:
            # Get the fixed adversarial answer
            adv_answer = adversarial_answers[self.id]
            if self.domain == "mmlu" or self.domain == "openbookqa":
                if adv_answer not in ["A", "B", "C", "D"]:
                    pdb.set_trace()
                    raise ValueError("Invalid fixed answer")
                fixed_answer = adv_answer
            elif self.domain == "gsm8k":
                fixed_answer = "\\boxed{" + adv_answer + "}"
            elif self.domain == "arc-c" or self.domain == "medqa":
                if adv_answer not in ["A", "B", "C", "D", "E"]:
                    pdb.set_trace()
                    raise ValueError("Invalid fixed answer")
                fixed_answer = adv_answer
            
            # Still generate analysis to make it look realistic
            tasks = [asyncio.create_task(self._async_execute(input, comm_info, adversarial_answer=adv_answer, **kwargs))]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            
            # Replace the answer part with the fixed adversarial answer
            for result in results:
                if not isinstance(result, list):
                    result = [result]
                for res in result:
                    # Extract analysis part (everything after first character which is the answer)
                    if len(res) > 1:
                        analysis = res.strip()

                        # Combine fixed answer with analysis
                        if self.domain == "gsm8k":
                            modified_result = f"{analysis}\n\nThe answer is {fixed_answer}\n\n{adv_add_on}"
                        elif self.domain == "mmlu" or self.domain == "arc-c" or self.domain == "medqa" or self.domain == "openbookqa":
                            modified_result = f"{fixed_answer}\n\n{analysis}\n\n{adv_add_on}"
                        else:
                            raise ValueError("Invalid domain: {self.domain}, node execution failed")
                    else:
                        # Just use the fixed answer if no analysis
                        # modified_result = fixed_answer
                        
                        # pdb.set_trace()
                        raise ValueError("No analysis found")
                    # pdb.set_trace()
                    self.outputs.append(modified_result)
        else:
            # Normal execution for non-adversarial agents
            # if len(comm_info) > 0:
            #     pdb.set_trace()
            tasks = [asyncio.create_task(self._async_execute(input, comm_info, **kwargs))]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            # pdb.set_trace()
            for result in results:
                if result is None:
                    pdb.set_trace()
                    raise ValueError(f"Node {self.id} (role={getattr(self, 'role', 'unknown')}) _async_execute returned None. Check LLM response.")
                if not isinstance(result, list):
                    result = [result]
                # Filter out any None values before extending
                result = [r for r in result if r is not None]
                if not result:
                    pdb.set_trace()
                    raise ValueError(f"Node {self.id} (role={getattr(self, 'role', 'unknown')}) produced empty/None results. Check LLM response.")
                self.outputs.extend(result)
        
        # Final safety check for None in outputs
        if any(o is None for o in self.outputs):
            pdb.set_trace()
            raise ValueError(f"Node {self.id} (role={getattr(self, 'role', 'unknown')}) produced None output. Check LLM response.")
        
        # if self.role == "Math Solver":
        #     question = input['question']
        #     response = self.outputs[-1]
        #     question_response = f"Question: {question}\nResponse: {response}"
        #     self.write_to_file(question_response, "math_solver_outputs.txt")
        
        if len(self.outputs) == 0:
            pdb.set_trace()
            raise ValueError("No outputs found")
        # pdb.set_trace()
        return self.outputs
    
    # async def async_execute(self, input:Any, adversarial_answers:Dict[str,str]={}, **kwargs):

    #     self.outputs = []
    #     comm_info:Dict[str,Dict] = self.get_comm_info()

    #     if len(comm_info) > 0:
    #         pdb.set_trace()
    #     tasks = [asyncio.create_task(self._async_execute(input, comm_info, **kwargs))]
    #     results = await asyncio.gather(*tasks, return_exceptions=False)
    #     # pdb.set_trace()
    #     for result in results:
    #         if not isinstance(result, list):
    #             result = [result]
    #         self.outputs.extend(result)
    #     # pdb.set_trace()
    #     return self.outputs
               
    @abstractmethod
    def _execute(self, input:List[Any], comm_info:Dict[str,Dict], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    async def _async_execute(self, input:List[Any], comm_info:Dict[str,Dict], adversarial_answer:str = None, **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    def _process_inputs(self, raw_inputs:List[Any], comm_info:Dict[str,Dict], adversarial_answer:str = None, **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
