import torch
from collections import namedtuple
import numpy as np
import torch_geometric.utils as U
from torch import tanh as tanh
from torch import sigmoid as sigmoid
from torch import softmax as softmax

'''
score_method: int / 'lin' 'softmax' 'tanh' 'sigmoid'
aggregate_method='avg', 'min', 'max' (score uased as node_score)
'''

class pooling(torch.nn.Module):
    def __init__(self, in_channels, score_method='lin', p1=0, p2=1, aggregate_score_method='avg'):
        super().__init__()
        self.p1 = p1
        self.p2 = p2 
        self.score_method = score_method
        self.aggregate_score_method = aggregate_score_method
        if type(score_method) not in [int, float]:
            self.lin = torch.nn.Linear(2*in_channels, 1)
   
    def forward(self, edge_index, x):
        edge_score = self.get_edge_score(edge_index=edge_index, x=x)
        node_min_score, node_max_score, node_avg_score = self.get_node_score(edge_score, edge_index)
        strict_dict, strict_clustered_nodes = self.get_strict_cluster(edge_score, node_min_score, edge_index) 
        full_dict, _, _ = self.get_soft_cluster(edge_score, edge_index, strict_dict, strict_clustered_nodes)
        self.test_completeness(full_dict, node_min_score)
        new_adjacency_matrix, new_x, node_assignment_matrix, score_assignment_matrix = self.get_next_layer(full_dict, aggregate_score=eval('node_'+self.aggregate_score_method+'_score'), edge_index=edge_index, x=x)
        new_edge_index= U.remove_self_loops(U.dense_to_sparse(new_adjacency_matrix)[0])[0]
        return new_edge_index, new_adjacency_matrix, new_x, score_assignment_matrix, edge_index # the last two are for unpooling
    

    def unpool(self, new_x, score_assignment_matrix, edge_index):
        x = torch.matmul(torch.nan_to_num(1/score_assignment_matrix.t(),nan=0.0, posinf=0.0, neginf=0.0), new_x)
        return x, edge_index

    # Below are functions used in forward
    def get_edge_score(self, edge_index=torch.tensor([]), x=torch.tensor([])):
        score_method = self.score_method
        if type(score_method) in [int, float]:
            pdist = torch.nn.PairwiseDistance(score_method)
            edge_score = pdist(x[edge_index[0]], x[edge_index[1]])
        else:
            lin = self.lin
            if score_method == 'softmax':
                e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1) + torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=-1)
                edge_score = softmax(lin(e).view(-1), dim=0)
            elif score_method == 'tanh':
                e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1) + torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=-1)
                edge_score = tanh(lin(e).view(-1))
            elif score_method == 'sigmoid':
                e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1) + torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=-1)
                edge_score = sigmoid(lin(e).view(-1))
            elif score_method == 'lin':
                e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1) + torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=-1)
                edge_score = lin(e).view(-1) 
            else:
                print('Wrong method input')
        return edge_score
    
    # project the score of egdes to nodes
    def get_node_score(self, edge_score=torch.tensor([]), edge_index=torch.tensor([])):
        num_nodes = torch.max(edge_index) + 1
        node_min_score = torch.tensor([])
        node_max_score = torch.tensor([])
        node_avg_score = torch.tensor([])
        node_score_dict = {}
        # Initialize dict 
        for i in range(num_nodes):
            node_score_dict[i] = torch.tensor([])
        for i in range(edge_score.shape[0]):
            node_score_dict[edge_index[0,i].item()] = torch.cat([node_score_dict[edge_index[0,i].item()], edge_score[i,None]], 0)
            node_score_dict[edge_index[1,i].item()] = torch.cat([node_score_dict[edge_index[1,i].item()], edge_score[i,None]], 0)
        for i in range(num_nodes):
            node_min_score = torch.cat([node_min_score, torch.min(node_score_dict[i], dim=0, keepdim=True).values], 0)
            node_max_score = torch.cat([node_max_score, torch.max(node_score_dict[i], dim=0, keepdim=True).values], 0)
            node_avg_score = torch.cat([node_avg_score, torch.mean(node_score_dict[i], dim=0, keepdim=True)], 0)
        return node_min_score, node_max_score, node_avg_score


    def get_strict_cluster(self, edge_score, node_min_score, edge_index):
        p1 = self.p1 
        p2 = self.p2
        edge_sep = torch.quantile(node_min_score, q=p1)
        edge_sep_upper = torch.quantile(node_min_score, q=p2)
        num_cluster = -1
        strict_dict = {}
        strict_clustered_nodes = torch.tensor([])
        # strict connected clusters
        for i in range(edge_score.shape[0]):
            # Edge that has a score lower than the lower_bound
            if edge_score[i].item() <= edge_sep:
                # Neither of the nodes have been assigned to a clsuter
                if (edge_index[0,i] not in strict_clustered_nodes) & (edge_index[1,i] not in strict_clustered_nodes):
                    strict_clustered_nodes = torch.cat([strict_clustered_nodes, edge_index[0,i,None], edge_index[1,i,None]], 0)
                    num_cluster += 1
                    strict_dict[num_cluster] = edge_index[:,i]
                # One of the node has been assigned to a cluster
                elif (edge_index[0,i]in strict_clustered_nodes) & (edge_index[1,i] not in strict_clustered_nodes):
                    strict_clustered_nodes = torch.cat([strict_clustered_nodes, edge_index[1,i,None]], 0)
                    for key, value in strict_dict.items():
                        if edge_index[0,i].item() in value:
                            cluster_ind = key
                    strict_dict[cluster_ind] = torch.concat([strict_dict[cluster_ind], edge_index[1,i,None]])
                elif (edge_index[0,i] not in strict_clustered_nodes) & (edge_index[1,i] in strict_clustered_nodes):
                    strict_clustered_nodes = torch.cat([strict_clustered_nodes, edge_index[0,i,None]], 0)
                    for key, value in strict_dict.items():
                        if edge_index[1,i].item() in value:
                            cluster_ind = key
                    strict_dict[cluster_ind] = torch.concat([strict_dict[cluster_ind], edge_index[0,i,None]])
                # The two nodes are assigned to different cluster, need to combine the two clusters into one cluster
                elif (edge_index[0,i] in strict_clustered_nodes) & (edge_index[1,i] in strict_clustered_nodes):
                    for key, value in strict_dict.items():
                        if edge_index[0,i].item() in value:
                            cluster_ind0 = key
                        if edge_index[1,i].item() in value:
                            cluster_ind1 = key
                    if cluster_ind0 != cluster_ind1:
                        strict_dict[cluster_ind0] = torch.unique(torch.concat([strict_dict[cluster_ind0], strict_dict[cluster_ind1]], 0))
                        strict_dict.pop(cluster_ind1)
        # strict independet clusters
        for i in range(node_min_score.shape[0]):
            if node_min_score[i] > edge_sep_upper:
                num_cluster += 1
                strict_dict[num_cluster] = torch.tensor([i])
                strict_clustered_nodes = torch.cat([strict_clustered_nodes, torch.tensor([i])], 0)
        return strict_dict, strict_clustered_nodes  


    def get_soft_cluster(self, edge_score, edge_index, strict_dict, strict_clustered_nodes):
        soft_dict = {}
        full_dict = {}
        for key, value in strict_dict.items():
            full_dict[key] = torch.clone(value)
        soft_clustered_nodes = torch.tensor([])
        num_cluster = -1
        # First, form the soft clusters
        for i in range(edge_score.shape[0]):
            # one node in strict cluster, the other node not in strict cluster, add a soft cluster if the node is not in any soft_cluster
            if (edge_index[0,i] not in strict_clustered_nodes) & (edge_index[1,i] in strict_clustered_nodes):
                if edge_index[0,i] not in soft_clustered_nodes:
                    num_cluster += 1
                    soft_dict[num_cluster] = edge_index[0,i,None]
                    soft_clustered_nodes = torch.cat([soft_clustered_nodes, edge_index[0,i,None]], 0)
            
            elif (edge_index[1,i] not in strict_clustered_nodes) & (edge_index[0,i] in strict_clustered_nodes):
                if edge_index[1,i] not in soft_clustered_nodes:
                    num_cluster += 1
                    soft_dict[num_cluster] = edge_index[1,i,None]
                    soft_clustered_nodes = torch.cat([soft_clustered_nodes, edge_index[1,i,None]], 0)

            # neither of the nodes are in strict_cluster
            elif (edge_index[0,i] not in strict_clustered_nodes) & (edge_index[1,i] not in strict_clustered_nodes):
                # neither of the nodes are in soft_cluster, add a new soft_cluster
                if (edge_index[0,i] not in soft_clustered_nodes) & (edge_index[1,i] not in soft_clustered_nodes):
                    num_cluster += 1
                    soft_dict[num_cluster] = edge_index[:,i]  
                    soft_clustered_nodes = torch.cat([soft_clustered_nodes, edge_index[0,i,None], edge_index[1,i,None]], 0)
                # one of the node is in soft_cluster, add the other node in the same cluster
                elif (edge_index[0,i] in soft_clustered_nodes) & (edge_index[1,i] not in soft_clustered_nodes):
                    for key, value in soft_dict.items():
                        if edge_index[0,i].item() in value:
                            cluster_ind = key
                    soft_dict[cluster_ind] = torch.cat([soft_dict[cluster_ind], edge_index[1,i,None]], 0)
                    soft_clustered_nodes = torch.cat([soft_clustered_nodes, edge_index[1,i,None]], 0)
                # The two nodes are in different soft_clusters, combine the two nodes together
                elif (edge_index[1,i] in soft_clustered_nodes) & (edge_index[0,i] not in soft_clustered_nodes):
                    soft_clustered_nodes = torch.cat([soft_clustered_nodes, edge_index[0,i,None]], 0)
                    for key, value in soft_dict.items():
                        if edge_index[1,i].item() in value:
                            cluster_ind = key
                    soft_dict[cluster_ind] = torch.cat([soft_dict[cluster_ind], edge_index[0,i,None]], 0)

                elif  (edge_index[0,i] in soft_clustered_nodes) & (edge_index[1,i] in soft_clustered_nodes):
                    for key, value in soft_dict.items():
                        if edge_index[0,i].item() in value:
                            cluster_ind0 = key
                        if edge_index[1,i].item() in value:
                            cluster_ind1 = key
                    if cluster_ind0 != cluster_ind1:
                        soft_dict[cluster_ind0] = torch.unique(torch.cat([soft_dict[cluster_ind0], soft_dict[cluster_ind1]], 0))
                        soft_dict.pop(cluster_ind1)
        # Second, assign soft clusters to strict clusters
        for i in range(edge_score.shape[0]):
            if (edge_index[0,i] in soft_clustered_nodes) & (edge_index[1,i] in strict_clustered_nodes):
                for key, value in strict_dict.items():
                    if edge_index[1,i].item() in value:
                        cluster_ind = key 
                for key, value in soft_dict.items():
                    if edge_index[0,i].item() in value:
                        full_dict[cluster_ind] = torch.unique(torch.cat([full_dict[cluster_ind], soft_dict[key]], 0))
            if (edge_index[1,i] in soft_clustered_nodes) & (edge_index[0,i] in strict_clustered_nodes):
                for key, value in strict_dict.items():
                    if edge_index[0,i].item() in value:
                        cluster_ind = key 
                for key, value in soft_dict.items():
                    if edge_index[1,i].item() in value:
                        full_dict[cluster_ind] = torch.unique(torch.cat([full_dict[cluster_ind], soft_dict[key]], 0))
        return full_dict, soft_dict, soft_clustered_nodes


    def test_completeness(self, full_dict, node_min_score):
        all_clustered_nodes = torch.tensor([])
        for key, value in full_dict.items():
            all_clustered_nodes = torch.unique(torch.cat([all_clustered_nodes, value], 0))
        for i in range(node_min_score.shape[0]):
            if (i in all_clustered_nodes) == False:
                print(f'{i:05d} is not included')


    def get_next_layer(self, full_dict, aggregate_score, edge_index=torch.tensor([]), x=torch.tensor([])):
        full_dict_keys = list(full_dict.keys())
        num_cluster = len(full_dict_keys)
        num_nodes = aggregate_score.shape[0]
        node_assignment_matrix = torch.zeros(num_cluster, num_nodes)
        score_assignment_matrix =  torch.zeros(num_cluster, num_nodes)
        new_adjacency_matrix = torch.zeros(num_cluster, num_cluster)
        
        new_x = torch.zeros(num_cluster, x.shape[1])
        for i in range(num_cluster):
            for j in full_dict[full_dict_keys[i]]:
                node_assignment_matrix[i,j] = 1
                score_assignment_matrix[i,j] = aggregate_score[j]
        # new_adjacency_matrix = torch.matmul(node_assignment_matrix,node_assignment_matrix.t())
        new_adjacency_matrix = torch.matmul(torch.matmul(node_assignment_matrix,U.to_dense_adj(edge_index)[0]),node_assignment_matrix.t()) 
        new_adjacency_matrix[new_adjacency_matrix>0] = 1
        new_x = torch.matmul(score_assignment_matrix, x)
        return new_adjacency_matrix, new_x, node_assignment_matrix, score_assignment_matrix