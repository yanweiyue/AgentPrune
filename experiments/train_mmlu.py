import torch
import torch.nn.functional as F
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy

from AgentPrune.graph.graph import Graph
from experiments.accuracy import Accuracy
from AgentPrune.utils.globals import Cost, PromptTokens, CompletionTokens
from AgentPrune.utils.utils import nuclear_norm,frobenius_norm

async def train(graph:Graph,
            dataset,
            num_iters:int=100,
            num_rounds:int=1,
            lr:float=0.1,
            batch_size:int = 4,
            imp_per_iters: int = 1,
            pruning_rate: float = 0.05,
            args=None,
            kwargs=None,
          ) -> None:
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record
    
    loader = infinite_data_loader()
    
    optimizer = torch.optim.Adam([graph.spatial_logits,graph.temporal_logits], lr=lr)    
    
    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80*'-')
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []
        add_losses = []
        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            
            spatial_matrix_train = realized_graph.spatial_logits.reshape((sum(args.agent_nums),sum(args.agent_nums)))
            temporal_matrix_train = realized_graph.temporal_logits.reshape((sum(args.agent_nums),sum(args.agent_nums)))
            spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
            temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
            loss_s = nuclear_norm(spatial_matrix_train)
            loss_t = nuclear_norm(temporal_matrix_train)
            frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
            frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
            add_loss = loss_s + loss_t + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
            input_dict = dataset.record_to_input(record)
            print(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)
            add_losses.append(add_loss)
            
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []
        
        for raw_answer, log_prob, add_loss, correct_answer in zip(raw_answers, log_probs, add_losses, correct_answers):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            utility = accuracy.get()
            utilities.append(utility)
            single_loss = - log_prob * utility
            loss_list.append(single_loss+add_loss)
            print(f"correct answer:{correct_answer}")
    
        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad() 
        total_loss.backward()
        optimizer.step()
        spatial_probs = torch.sigmoid(graph.spatial_logits)
        temporal_probs = torch.sigmoid(graph.temporal_logits)
        
        print("raw_answers:",raw_answers)
        print("answers:",answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities)
        print("loss:", total_loss.item()) 
        print("Spatial logits Grad:", graph.spatial_logits.grad)
        print("Temporal logits Grad:", graph.spatial_logits.grad)
        print("Spatial logits:", graph.spatial_logits)
        print("Temporal logits:", graph.temporal_logits)
        print("Spatial probs:", spatial_probs)
        print("Temporal probs:", temporal_probs)
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
        
        if (i_iter+1)%imp_per_iters == 0:
            spatial_masks, temporal_masks = graph.update_masks(pruning_rate)
            print("spatial masks:",spatial_masks)
            print("temporal masks:",temporal_masks)
            print("spatial sparsity:",spatial_masks.sum()/spatial_masks.numel())
            print("temporal sparsity:",temporal_masks.sum()/temporal_masks.numel())
        