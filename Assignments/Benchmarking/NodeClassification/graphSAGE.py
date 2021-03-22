"""
Reference:
https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/graph_sage.py
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear
#from torch_geometric.nn import SAGEConv
from SAGE_opt import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, aggr='mean'):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden, aggr)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden, aggr))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
        # print('SAGE')

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, dataX, dataY):
        activation = F.relu  # torch.sigmoid
        x, edge_index = dataX, dataY
        x = activation(self.conv1(x, edge_index))
        for conv in self.convs:
            x = activation(conv(x, edge_index))
        #x = global_mean_pool(x, batch)
        x = activation(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
