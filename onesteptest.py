import torch
from model.net import EncoderRNN
import numpy as np

batch_size = 3
seq_length = 5
hidden_size = 5
vocabulary_size = 30

net = EncoderRNN(input_vocabulary_size=vocabulary_size, batch_size=batch_size, hidden_size=hidden_size)

test_batch = torch.tensor(np.arange(15).reshape(seq_length, batch_size), dtype=torch.long)
test_init_hidden_state = net.init_hidden(device='cpu')

output, hidden = net(test_batch, test_init_hidden_state)
# print(output)
print(output.size())
# print(hidden)
print(hidden.size())

onestep_hidden = hidden
onestep_output = output

hidden = test_init_hidden_state
hts = []
for i in range(seq_length):
    output, hidden = net(test_batch[i], hidden)
    print(output.equal(hidden))
    hts.append(output)
hts = torch.cat(hts)
print(hts[-1])
print(hidden)
print(hts.size())
# print(hts)
print(hts.equal(onestep_output))
print(hidden.equal(onestep_hidden))
