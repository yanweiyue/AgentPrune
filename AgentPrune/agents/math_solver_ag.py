from typing import List,Any,Dict

from AgentPrune.graph.autogen_node import NodeAutoGen
from AgentPrune.agents.agent_registry import AgentRegistry
from AgentPrune.llm.llm_registry import LLMRegistry
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry
from AgentPrune.tools.coding.python_executor import execute_code_get_return
from dataset.gsm8k_dataset import gsm_get_predict

@AgentRegistry.register('MathSolverAG')
class MathSolverAG(NodeAutoGen):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "",):
        super().__init__(id, "MathSolverAG" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role) 
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """             
        system_prompt = self.constraint
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"], role=self.role)
        if self.role == "Math Solver":
            user_prompt += "(Hint: The answer is near to"
            for message in self.conversation_history:
                if message['role'] == 'assistant':
                    user_prompt += " "+gsm_get_predict(message['content'])
            user_prompt += ")."
        user_prompt += "Please continue to complete the task based on the previous conversation."
        return system_prompt, user_prompt
    
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        history_conversation = self.get_neighbor_history(input['task'])
        message = history_conversation + [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        if self.role == "Programming Expert":
            answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
            response += f"\nthe answer is {answer}" if answer is not None and "Error occurred" not in response else ""
        self.update_conversation_history(response,history_conversation)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        history_conversation = self.get_neighbor_history(input['task'])
        message = history_conversation + [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        if self.role == "Programming Expert":
            answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
            response += f"\nthe answer is {answer}" if answer is not None and "Error occurred" not in response else ""
        self.update_conversation_history(response,history_conversation)
        return response