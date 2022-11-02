import torch
from torch import nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_layer, n_class, bidirectional=False):
        super(LSTM, self).__init__()

        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True, bidirectional=bidirectional, dropout=0.1)
        self.softmax = nn.Softmax(dim=-1)

        if self.bidirectional:
            self.Classifier = nn.Linear(hidden_dim*2, n_class)

        else:
            self.Classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        if self.bidirectional:
            out = torch.hstack((hn[-2, :, :], hn[-1, :, :]))
        else:
            out = out[:, -1, :]
        out = self.Classifier(out)
        return self.softmax(out)


class LSTM_ss(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_layer, output=1, bidirectional=False):
        super(LSTM_ss, self).__init__()

        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True, bidirectional=bidirectional, dropout=0.1)


        if self.bidirectional:
            self.Classifier1 = nn.Linear(hidden_dim*2, 64)
            self.Classifier = nn.Linear(64, output)

        else:
            self.Classifier1 = nn.Linear(hidden_dim, 64)
            self.Classifier = nn.Linear(64, output)

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        if self.bidirectional:
            out = torch.hstack((hn[-2, :, :], hn[-1, :, :]))
        else:
            out = out[:, -1, :]
        out = self.Classifier1(out)
        out = self.Classifier(out)
        return out
