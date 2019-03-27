import unicodedata
import os
import re
import random
import argparse

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2

    def add_sentence(self, sentence: str):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1


def unicode2ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'M')


def normalize_string(s: str):
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s


def read_languages(lang1, lang2, reverse=False):
    # eng to fra
    print('reading lines...')
    lines = open('./data/fra-eng/fra.txt').readlines()

    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Language(lang2)
        output_lang = Language(lang1)
    else:
        input_lang = Language(lang1)
        output_lang = Language(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_languages(lang1, lang2, reverse)
    print('read {} sentence pairs'.format(len(pairs)))
    pairs = filter_pairs(pairs)
    print('trimmed to {} sentence pairs'.format(len(pairs)))
    print('counting words...')
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print('words counting done')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepare_data('eng', 'fra', reverse=True)
print(random.choice(pairs))


class EncoderRNN(nn.Module):
    def __init__(self, input_vocabulary_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_vocabulary_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embeded = self.embedding(input).view(1, 1, -1)
        output = embeded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_vocabulary_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_vocabulary_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_vocabulary_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexes_from_sentence(lang: Language, sentence: str):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang: Language, sentence: str):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensor_from_pair(pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    output_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, output_tensor


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder: EncoderRNN, decoder: DecoderRNN, encoder_optimizer: optim.Optimizer,
          decoder_optimizer: optim.Optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


from tqdm import tqdm


class RunningAverage:
    def __init__(self):
        self.counts = 0
        self.total_sum = 0
        self.avg = 0

    def reset(self):
        self.counts = 0
        self.total_sum = 0

    def update(self, val):
        self.total_sum += val
        self.counts += 1
        self.avg = self.total_sum / self.counts


def train_iters(encoder: EncoderRNN, decoder: DecoderRNN,
                n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    loss_avg = RunningAverage()
    current_best_loss = 1e3

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensor_from_pair(random.choice(pairs)) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    with tqdm(total=n_iters) as progress_bar:
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            loss_avg.update(loss)

            if iter % print_every == 0:
                tqdm.write('# of iterations: {}, loss average: {:.3f}'.format(iter, loss_avg.avg))
                if loss_avg.avg < current_best_loss:
                    tqdm.write('new best loss average found, saving model...')
                    current_best_loss = loss_avg.avg
                    state = {
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        'iters': iter
                    }
                    os.makedirs('./ckpts', exist_ok=True)
                    torch.save(state, './ckpts/best.pth.tar')
                loss_avg.reset()
            # if iter % plot_every == 0:
            #     plot_loss_avg = plot_loss_total / plot_every
            #     plot_losses.append(plot_loss_avg)
            #     plot_loss_total = 0

            # show_plot(plot_losses)
            progress_bar.set_postfix(loss_avg=loss_avg.avg)
            progress_bar.update()


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)

train_iters(encoder1, decoder1, 75000, print_every=100)
