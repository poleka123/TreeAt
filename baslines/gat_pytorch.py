import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        # e = Wh1 + Wh2.T
        e = Wh1 + Wh2.permute(0, 2, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, nclass, dropout, alpha, nheads, adj):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.adj = adj
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.ln = nn.LayerNorm([num_nodes, nhid])

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.output = OutputLayer(nclass, num_nodes)
        self.fc = nn.Linear(nclass, nclass)
    def forward(self, x):
        batch, node, time, channel = x.shape
        x = x.reshape(batch, node, time*channel)
        x = F.dropout(x, self.dropout, training=self.training) #x shape is (B, N, T, C)
        x = torch.cat([self.ln(att(x, self.adj)) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        # return self.fc(x)
        return self.output(x)
        # return F.log_softmax(x, dim=1)

class OutputLayer(nn.Module):
    def __init__(self, c, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv1d(c, c, 1, 1, dilation=1, padding=0)
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv1d(c, c, 1, 1, dilation=1, padding=0)
        self.fc = nn.Linear(c, c)

    def forward(self, x):
        # x_t1 = self.tconv1(x.permute(0, 2, 1))
        # x_ln = self.ln(x_t1.permute(0, 2, 1)).permute(0, 2, 1)
        x_ln = self.ln(x)
        # x_t2 = self.tconv2(x_ln)
        return x_ln
        # return self.fc(x_ln)
        # return self.fc(x_ln.permute(0, 2, 1))