import torch
import random
import argparse
import os
import logging

from model.net import EncoderRNN, DecoderRNN
from build_dataset import MAX_LENGTH, SOS_token, EOS_token
from build_dataset import prepare_data, Language
from utils import tensor_from_sentence
from utils import Params, set_logger


def evaluate(encoder: EncoderRNN,
             decoder: DecoderRNN,
             input_lang: Language,
             output_lang: Language,
             sentence,
             device,
             max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence).to(device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden(device)

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


def evaluate_randomly(encoder: EncoderRNN,
                      deocder: DecoderRNN,
                      input_lang: Language,
                      output_lang: Language,
                      pairs,
                      device,
                      n=10):
    for i in range(n):
        pair = random.choice(pairs)
        logging.info('> ' + pair[0])
        logging.info('= ' + pair[1])
        output_words = evaluate(encoder, deocder, input_lang, output_lang, pair[0], device)
        output_sentence = ' '.join(output_words)
        logging.info('< ' + output_sentence)
        logging.info('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--model_dir', default='experiments/base_model', type=str)
    args = parser.parse_args()

    params_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.exists(params_path), 'no json configuration file was found at {}'.format(params_path)
    hps = Params(params_path)
    args.__dict__.update(hps.dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    set_logger(os.path.join(args.model_dir, 'eval.log'), terminal=True)

    input_lang, output_lang, pairs = prepare_data('eng', 'fra', reverse=True)
    encoder = EncoderRNN(input_lang.n_words, args.hidden_size).to(device)
    decoder = DecoderRNN(args.hidden_size, output_lang.n_words).to(device)
    print('loading model...')
    state = torch.load(os.path.join(args.model_dir, 'ckpts', 'best.pth.tar'))
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    print('model loaded, start translating...')
    evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs, device)
