from typing import List,Any,Dict
import re

from AgentPrune.graph.autogen_node import NodeAutoGen
from AgentPrune.agents.agent_registry import AgentRegistry
from AgentPrune.llm.llm_registry import LLMRegistry
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry
from AgentPrune.tools.search.wiki import search_wiki_main

def find_strings_between_pluses(text):
    return re.findall(r'\@(.*?)\@', text)

@AgentRegistry.register('AnalyzeAgentAG')
class AnalyzeAgentAG(NodeAutoGen):
    def __init__(self, id: str | None =None, role:str = None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "AnalyzeAgentAG" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        
    async def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """              
        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n" if self.role != 'Fake' else self.prompt_set.get_adversarial_answer_prompt(raw_inputs['task'])
        user_prompt += "Please continue to complete this task based on the previous conversation."
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        history_conversation = self.get_neighbor_history(input['task'])
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = history_conversation + [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        self.update_conversation_history(response,history_conversation)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        history_conversation = self.get_neighbor_history(input['task'])
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)
        message = history_conversation + [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        self.update_conversation_history(response,history_conversation)
        return response