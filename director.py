import os
import torch

from modules.encoder.encoder import EncoderRNN
from modules.data_loader.data_loader import DataLoaderProducer
from modules.decoder.vanilla import DecoderRNN
from modules.decoder.attention import AttnDecoderRNN


class Director:
    def __init__(self, hps):
        os.environ['CUDA_VISIBLE_DEVICES'] = hps.gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hps = hps

        dl_producer = DataLoaderProducer(max_length=hps.max_length, data_dir=hps.data_dir, mode=hps.mode)
        self.src_language, self.tgt_language, self.dl = dl_producer.prepare_data_loader(batch_size=hps.batch_size,
                                                                                        num_workers=hps.num_workers)
        self.encoder = EncoderRNN(batch_size=hps.batch_size, input_vocabulary_size=self.src_language.n_words,
                                  hidden_size=hps.hidden_size)
        self.decoder = self.get_decoder()

    def get_decoder(self):
        if self.hps.decoder_type == 'attn':
            return AttnDecoderRNN(output_vocabulary_size=self.tgt_language.n_words, batch_size=self.hps.batch_size,
                                  hidden_size=self.hps.hidden_size, dropout_rate=self.hps.dropout_rate,
                                  max_length=self.hps.max_length)
        elif self.hps.decoder_type == 'vanilla':
            return DecoderRNN(output_vocabulary_size=self.tgt_language.n_words, batch_size=self.hps.batch_size,
                              hidden_size=self.hps.hidden_size)
        else:
            raise ValueError('decoder type is illegal')
