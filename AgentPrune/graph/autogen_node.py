from typing import List, Dict, Optional, Any
import asyncio
from AgentPrune.graph.node import Node

class NodeAutoGen(Node):
    def __init__(self, id: Optional[str], agent_name:str="", domain:str="", llm_name:str = "",):
        super().__init__(id, agent_name, domain, llm_name)
        self.conversation_history : List[Dict] = [] # chat history of the whole conversation

    def spatial_or_temporal(self):
        if len(self.spatial_predecessors) and len(self.temporal_predecessors) == 0:
            return 'spatial'
        elif len(self.spatial_predecessors) == 0 and len(self.temporal_predecessors):
            return 'temporal'
        elif len(self.spatial_predecessors) == 0 and len(self.temporal_predecessors) == 0:
            return 'none'
        else:
            return 'both'
        
    def get_neighbor_history(self,input:str)->List[Dict]:
        neighbor_type = self.spatial_or_temporal()
        if neighbor_type == 'spatial':
            neighbor = self.spatial_predecessors[0]
        elif neighbor_type == 'temporal':
            neighbor = self.temporal_predecessors[0]
        elif neighbor_type == 'both':
            neighbor = self.temporal_predecessors[0]
        else:
            neighbor = None
        if neighbor is not None:
            conversation_history = neighbor.conversation_history
        else:
            conversation_history = []
        if len(conversation_history) == 0:
            conversation_history = [{'role':'user','content':input}]
        return conversation_history
        
    def update_conversation_history(self, content:str, neighbor_history:List[Dict]=[]):
        self.conversation_history = neighbor_history + [{'role':'assistant','content':content}]
