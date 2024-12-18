# In summary, the UV_Aggregator class is designed for aggregating embeddings of neighbors 
# in a graph-based machine learning model, and its behavior is determined by the uv flag,
# which specifies whether user or item aggregation is performed. 
# This is commonly used in graph-based neural network models for recommendation systems and other applications.

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention


class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_uv, history_r):

        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_uv)):
            history = history_uv[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]



            if self.uv == True:
                # user component
                e_uv = self.v2e.weight[history]
                uv_rep = self.u2e.weight[nodes[i]]
            else:
                # item component
                e_uv = self.u2e.weight[history]
                uv_rep = self.v2e.weight[nodes[i]]

            e_r = self.r2e.weight[tmp_label]
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            # Remove the item attention
            #att_history = torch.mean(o_history, dim=0)  # Directly sum or average the embeddings
            #embed_matrix[i] = att_history

            att_w = self.att(o_history, uv_rep, num_histroy_item)  ### 3 lines from here uncomment -- this is to remove author(user) attention 
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history  ## uncomment 
            # embed_matrix[i] = o_history #comment Author attention
        to_feats = embed_matrix
        return to_feats


# class UV_Aggregator(nn.Module):
#     """
#     item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
#     """

#     def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
#         super(UV_Aggregator, self).__init__()
#         self.uv = uv
#         self.v2e = v2e
#         self.r2e = r2e
#         self.u2e = u2e
#         self.device = cuda
#         self.embed_dim = embed_dim
#         self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
#         self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
#         self.att = Attention(self.embed_dim)

#     def forward(self, nodes, history_uv, history_r):

#         embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

#         for i in range(len(history_uv)):
#             history = history_uv[i]

#             if self.uv == True:
#                 # user component
#                 e_uv = self.v2e.weight[history]
#                 #uv_rep = self.u2e.weight[nodes[i]]
#             else:
#                 # item component
#                 e_uv = self.u2e.weight[history]
#                 #uv_rep = self.v2e.weight[nodes[i]]
#             agg_history = torch.sum(e_uv, dim=0)
#             embed_matrix[i] = agg_history

#         to_feats = embed_matrix
#         return to_feats

    


