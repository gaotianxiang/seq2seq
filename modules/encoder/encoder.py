import torch.nn as nn
import torch


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
