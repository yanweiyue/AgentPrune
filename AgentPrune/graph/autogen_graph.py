from typing import List,Any,Dict,Optional

import torch
from AgentPrune.graph.graph import Graph

class GraphAutoGen(Graph):
    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,):
        super().__init__(domain,llm_name,agent_names,decision_method,
                         optimized_spatial,initial_spatial_probability,fixed_spatial_masks,
                         optimized_temporal,initial_temporal_probability,fixed_temporal_masks,node_kwargs)
        self.chain_idx_list:List[int] = []
        self.chain_str_list:List[str] = []

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        last_node_idx = 0
        for i, in_node in enumerate(self.nodes.keys()):
            if i == 0:
                self.chain_idx_list.append(i)
                self.chain_str_list.append(in_node)
            if i < last_node_idx:
                continue
            for j, out_node in enumerate(self.nodes.keys()):
                if i >= j:
                    continue
                edge_idx = i*len(self.nodes)+j
                edge_prob = torch.sigmoid(self.spatial_logits[edge_idx] / temperature)
                if torch.rand(1) < edge_prob:
                    self.nodes[in_node].add_successor(self.nodes[out_node],'spatial')
                    log_probs.append(torch.log(edge_prob))
                    last_node_idx = j
                    self.chain_idx_list.append(j)
                    self.chain_str_list.append(out_node)
                    break
                else:
                    log_probs.append(torch.log(1 - edge_prob))
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))
        first_agent_idx:int = self.chain_idx_list[0]
        last_agent_idx:int = self.chain_idx_list[-1]
        first_agent:str = self.chain_str_list[0]
        last_agent:str = self.chain_str_list[-1]

        edge_idx = first_agent_idx*len(self.nodes)+last_agent_idx
        edge_prob = torch.sigmoid(self.temporal_logits[edge_idx] / temperature)
        if torch.rand(1) < edge_prob:
            self.nodes[first_agent].add_successor(self.nodes[last_agent],'temporal')
            log_probs.append(torch.log(edge_prob))
        else:
            log_probs.append(torch.log(1 - edge_prob))
        return torch.sum(torch.stack(log_probs))
