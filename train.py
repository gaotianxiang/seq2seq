import os
import random
import argparse

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm, trange
from utils import RunningAverage
from model.net import EncoderRNN, DecoderRNN
from build_dataset import prepare_data, EOS_token, SOS_token, MAX_LENGTH
from utils import tensor_from_sentence

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_lang, output_lang, pairs = prepare_data('eng', 'fra', reverse=True)
print(random.choice(pairs))

from torch.utils import data as data

from model.data_loader import Fra2Eng


def tensor_from_pair(pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0], device=device)
    output_tensor = tensor_from_sentence(output_lang, pair[1], device=device)
    return input_tensor, output_tensor


pairs = [tensor_from_pair(pair) for pair in pairs]
pairs = Fra2Eng(pairs)

pairs = data.DataLoader(pairs, batch_size=1, shuffle=True, num_workers=0)

teacher_forcing_ratio = 0.5


def train(input_tensor,
          target_tensor,
          encoder: EncoderRNN,
          decoder: DecoderRNN,
          encoder_optimizer: optim.Optimizer,
          decoder_optimizer: optim.Optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden(device)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

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


def train_iters(encoder: EncoderRNN,
                decoder: DecoderRNN,
                epochs,
                pairs,
                print_every=1000,
                learning_rate=0.01):
    loss_avg = RunningAverage()
    current_best_loss = 1e3

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = pairs

    criterion = nn.NLLLoss()

    for epoch in trange(epochs, desc='epochs'):
        i = 0
        with tqdm(total=len(training_pairs)) as progress_bar:
            for input_tensor, target_tensor in training_pairs:
                input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
                loss = train(input_tensor[0], target_tensor[0], encoder, decoder, encoder_optimizer, decoder_optimizer,
                             criterion)
                loss_avg.update(loss)
                i += 1
                if i % print_every == 0:
                    tqdm.write('# of iterations: {}, loss average: {:.3f}'.format(i, loss_avg.avg))
                    if loss_avg.avg < current_best_loss:
                        tqdm.write('new best loss average found, saving model...')
                        current_best_loss = loss_avg.avg
                        state = {
                            'encoder': encoder.state_dict(),
                            'decoder': decoder.state_dict(),
                            'iters': i,
                            'epoch': epoch
                        }
                        os.makedirs('./ckpts', exist_ok=True)
                        torch.save(state, './ckpts/best.pth.tar')
                    loss_avg.reset()

                progress_bar.set_postfix(loss_avg=loss_avg.avg)
                progress_bar.update()


if __name__ == '__main__':
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)

    train_iters(encoder1, decoder1, 100, pairs, print_every=100)
