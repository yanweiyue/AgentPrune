from typing import List,Any,Dict

from AgentPrune.graph.autogen_node import NodeAutoGen
from AgentPrune.agents.agent_registry import AgentRegistry
from AgentPrune.llm.llm_registry import LLMRegistry
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry
from AgentPrune.tools.coding.python_executor import PyExecutor

@AgentRegistry.register('FinalWriteCodeAG')
class FinalWriteCodeAG(NodeAutoGen):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "FinalWriteCodeAG" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results
    
    def _process_inputs(self, input:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()          
        system_prompt = f"{self.role}.\n {self.constraint}"
        user_prompt = f"The task is: {input['task']}\n"
        history_conversation = self.get_neighbor_history(input['task'])
        for i, message in enumerate(history_conversation):
            output = message['content']
            if output.startswith("```python") and output.endswith("```"):
                output = output.lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                user_prompt += f"The code written by the previous agent {i} is:\n\n{output}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
        user_prompt += "Please continue to complete this task based on the previous conversation."
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        self.internal_tests = self.extract_example(input)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        history_conversation = self.get_neighbor_history(input['task'])
        message = history_conversation + [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        self.internal_tests = self.extract_example(input)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        history_conversation = self.get_neighbor_history(input['task'])
        message = history_conversation + [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        return response


@AgentRegistry.register('FinalReferAG')
class FinalReferAG(NodeAutoGen):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "FinalRefer" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()          
        system_prompt = f"{self.role}.\n {self.constraint}"
        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = f"{decision_few_shot} The task is:\n\n {raw_inputs['task']}."
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        history_conversation = self.get_neighbor_history(input['task'])
        message = history_conversation + [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        history_conversation = self.get_neighbor_history(input['task'])
        message = history_conversation + [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        return response
