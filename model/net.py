import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_vocabulary_size, batch_size, hidden_size):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_vocabulary_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embeded = self.embedding(input).view(-1, self.batch_size, self.hidden_size)
        output = embeded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, device):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, output_vocabulary_size, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.embedding = nn.Embedding(output_vocabulary_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_vocabulary_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, self.batch_size, self.hidden_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
