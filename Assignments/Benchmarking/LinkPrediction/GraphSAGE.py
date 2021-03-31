import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, aggr, num_layers):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggr)
        # self.convs = torch.nn.ModuleList()
        # for i in range(num_layers - 2):
        #     self.convs.append(SAGEConv(h_feats, h_feats, aggr))
        self.conv2 = SAGEConv(h_feats, h_feats, aggr)
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x.float(), h_dst.float()))
        h = F.relu(h)
        # i = 1
        # for conv in self.convs:
        #     h_dst = h[:mfgs[i].num_dst_nodes()]
        #     h = conv(mfgs[i], (h, h_dst))
        #     h = F.relu(h)
        #     i += 1
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h
