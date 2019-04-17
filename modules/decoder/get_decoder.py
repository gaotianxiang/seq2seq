from .vanilla import DecoderRNN
from .attention import AttnDecoderRNN


def get_decoder(hps):
    if hps.mode == 'vanilla':
        return DecoderRNN(outpu)