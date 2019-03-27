import torch
import random

from model.net import EncoderRNN, DecoderRNN
from build_dataset import MAX_LENGTH, SOS_token, EOS_token
from main import tensor_from_sentence, input_lang, device, output_lang, pairs, encoder1, decoder1


def evaluate(encoder: EncoderRNN, decoder: DecoderRNN, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden(device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluate_randomly(encoder: EncoderRNN, deocder: DecoderRNN, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, deocder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


print('loading model...')
state = torch.load('./ckpts/best.pth.tar', map_location='cpu')
encoder1.load_state_dict(state['encoder'])
decoder1.load_state_dict(state['decoder'])
print('model loaded, start translating...')
evaluate_randomly(encoder1, decoder1)
