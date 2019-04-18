import os
import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
import random

from modules.encoder.encoder import EncoderRNN
from modules.data_loader.data_loader import DataLoaderProducer
from modules.data_loader.utils import SpecialToken
from modules.decoder.vanilla import DecoderRNN
from modules.decoder.attention import AttnDecoderRNN
from modules.data_loader.utils import tensor_from_sentence, add_padding

from utils import set_logger, log, RunningAverage


class Director:
    def __init__(self, hps):
        os.environ['CUDA_VISIBLE_DEVICES'] = hps.gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hps = hps

        self.dl_producer = DataLoaderProducer(max_length=hps.max_length, data_dir=hps.data_dir, mode=hps.s2t)
        self.src_language, self.tgt_language, self.dl = self.dl_producer.prepare_data_loader(batch_size=hps.batch_size,
                                                                                             num_workers=hps.num_workers)
        self.encoder = EncoderRNN(batch_size=hps.batch_size, input_vocabulary_size=self.src_language.n_words,
                                  hidden_size=hps.hidden_size).to(self.device)
        self.decoder = self.get_decoder().to(self.device)
        self.global_step = 0

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

    def train(self):
        log_dir = os.path.join(self.hps.model_dir, 'logs')
        ckpt_dir = os.path.join(self.hps.model_dir, 'ckpts')
        summ_dir = os.path.join(self.hps.model_dir, 'summary')
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(summ_dir, exist_ok=True)
        set_logger(os.path.join(log_dir, 'train.log'), terminal=False)

        epochs = self.hps.num_epochs
        print_every = self.hps.print_every
        log_every = self.hps.log_summary_every
        lr = self.hps.learning_rate

        loss_avg = RunningAverage()
        summary_writer = SummaryWriter(log_dir=summ_dir)
        current_best_loss = 1e3

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)

        training_pairs = self.dl

        criterion = nn.NLLLoss(reduce=False)

        if self.hps.resume:
            log('- load ckpts...')
            self.load_state_dict()

        for epoch in trange(epochs, desc='epochs'):
            loss_avg.reset()
            with tqdm(total=len(training_pairs)) as progress_bar:
                for language_pair, mask_pair in training_pairs:
                    language_pair, mask_pair = language_pair.to(self.device), mask_pair.to(self.device)
                    loss = self.train_single(language_pair, mask_pair, encoder_optimizer,
                                             decoder_optimizer, criterion)
                    loss_avg.update(loss.item())
                    self.global_step += 1
                    if self.global_step % log_every == 0:
                        summary_writer.add_scalar('loss_value', loss, global_step=self.global_step)
                    if self.global_step % print_every == 0:
                        log('global step: {}, loss average: {:.3f}'.format(self.global_step, loss_avg()))

                    progress_bar.set_postfix(loss_avg=loss_avg())
                    progress_bar.update()
            if loss_avg() < current_best_loss:
                log('new best loss average found, saving modules...')
                current_best_loss = loss_avg()
                state = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'global_step': self.global_step,
                    'epoch': epoch,
                    'loss_avg': loss_avg()
                }
                torch.save(state, os.path.join(ckpt_dir, 'best.pth.tar'))

    def train_single(self,
                     language_pair,
                     mask_pair,
                     encoder_optimizer: optim.Optimizer,
                     decoder_optimizer: optim.Optimizer,
                     criterion):
        input_tensors = language_pair[:, 0, :].t()
        target_tensors = language_pair[:, 1, :].t()

        target_masks = mask_pair[:, 1, :].t()

        encoder_init_hidden = self.encoder.init_hidden(self.device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        sentence_length = self.hps.max_length

        loss = 0

        encoder_output, encoder_final_hidden = self.encoder(input_tensors, encoder_init_hidden)

        decoder_input = torch.tensor([SpecialToken.SOS_token] * self.hps.batch_size, device=self.device)

        decoder_hidden = encoder_final_hidden

        use_teacher_forcing = True if random.random() < self.hps.teacher_forcing_ratio else False

        if use_teacher_forcing:
            for di in range(sentence_length):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
                loss += torch.sum(criterion(decoder_output, target_tensors[di]) * target_masks[di])
                decoder_input = target_tensors[di]
        else:
            for di in range(sentence_length):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += torch.sum(criterion(decoder_output, target_tensors[di]) * target_masks[di])

        loss = loss / target_masks.sum()
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss

    def load_state_dict(self):
        ckpt_path = os.path.join(self.hps.model_dir, 'ckpts', 'best.pth.tar')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError('- no ckpt file found')
        state_dict = torch.load(ckpt_path)
        self.global_step = state_dict['global_step']
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])
        log('- load ckpts from global step {}'.format(self.global_step))

    def test(self):
        log_dir = os.path.join(self.hps.model_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        set_logger(os.path.join(log_dir, 'test.log'), terminal=False)
        self.load_state_dict()
        src_language, tgt_language, pairs = self.dl_producer.prepare_data()

        for i in range(self.hps.sample_num):
            pair = random.choice(pairs)
            log('> ' + pair[0])
            log('= ' + pair[1])
            output_words = self.evaluate(src_language, tgt_language, pair[0])
            output_sentence = ' '.join(output_words)
            log('< ' + output_sentence)
            log('')

    def evaluate(self, input_lang, output_lang, sentence):
        with torch.no_grad():
            input_tensor = tensor_from_sentence(input_lang, sentence)
            input_tensor, _ = add_padding(input_tensor, self.hps.max_length)
            input_tensor = torch.tensor(input_tensor, dtype=torch.long).to(self.device)
            encoder_init_hidden = self.encoder.init_hidden(self.device)

            encoder_output, encoder_final_hidden = self.encoder(input_tensor, encoder_init_hidden)

            decoder_input = torch.tensor([[SpecialToken.SOS_token]], device=self.device)

            decoder_hidden = encoder_final_hidden

            decoded_words = []

            for di in range(self.hps.max_length):
                decoder_output, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden,
                                                                            encoder_output)
                topv, topi = decoder_output.topk(1)
                if topi.item() == SpecialToken.EOS_token:
                    # decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()

            return decoded_words
