from typing import List,Any,Dict

from AgentPrune.graph.autogen_node import NodeAutoGen
from AgentPrune.agents.agent_registry import AgentRegistry
from AgentPrune.llm.llm_registry import LLMRegistry
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry
from AgentPrune.tools.coding.python_executor import PyExecutor

@AgentRegistry.register('CodeWritingAG')
class CodeWritingAG(NodeAutoGen):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "",):
        super().__init__(id, "CodeWritingAG" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role) 
        
    def _process_inputs(self, input:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """             
        system_prompt = self.constraint
        user_prompt = f"The task is: {input['task']}\n"
        history_conversation = self.get_neighbor_history(input['task'])
        for i, message in enumerate(history_conversation):
            output = message['content']
            if output.startswith("```python") and output.endswith("```"):
                output = output.lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                if is_solved and len(self.internal_tests):
                    return "is_solved", output
                if i == len(history_conversation) - 1:
                    user_prompt += f"The code written by the previous agent is:\n\n{output}\n\n The code failed the test and the feedback was\n\n {feedback}.\n\n"
        user_prompt += "Please continue to complete this task based on the previous conversation."
        return system_prompt, user_prompt

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
    
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        self.internal_tests = self.extract_example(input)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        history_conversation = self.get_neighbor_history(input['task'])
        if system_prompt == "is_solved":
            self.update_conversation_history(user_prompt,history_conversation)
            return user_prompt
        message = history_conversation + [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        self.update_conversation_history(response,history_conversation)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        self.internal_tests = self.extract_example(input)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        history_conversation = self.get_neighbor_history(input['task'])
        if system_prompt == "is_solved":
            self.update_conversation_history(user_prompt,history_conversation)
            return user_prompt
        message = history_conversation + [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        self.update_conversation_history(response,history_conversation)
        return response