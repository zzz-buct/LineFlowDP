import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, GCNConv, global_mean_pool


class MyRGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_layers=3, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))

    def forward(self, x, edge_index, edge_type, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i != len(self.convs) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return x


class MyGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return x
