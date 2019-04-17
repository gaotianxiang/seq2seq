import random
import torch
import numpy as np
import os
import pickle

from torch.utils import data as data
from .utils import normalize_string, Language, SpecialToken, tensor_from_pair, add_padding_pairs


class DataLoaderProducer:
    def __init__(self, max_length, data_dir, mode='e2f'):
        self.max_length = max_length
        self.data_dir = data_dir
        self.mode = mode
        # self.src_language, self.tgt_language, self.pairs = self.read_dataset()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in ['e2f', 'f2e']:
            raise ValueError('mode has to be one of e2f or f2e')
        self._mode = mode

    def read_dataset(self):
        cache_path = os.path.join(self.data_dir, 'fra-eng.preprocess')
        if os.path.exists(cache_path):
            print('cache hit, read from cache...')
            pkl = pickle.load(open(cache_path, 'rb'))
            return pkl['src_language'], pkl['tgt_language'], pkl['pairs']
        print('cache miss...')
        print('reading lines...')
        lines = open(os.path.join(self.data_dir, 'fra.txt')).readlines()

        pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

        if self.mode == 'f2e':
            pairs = [list(reversed(p)) for p in pairs]
            src_language = Language('French')
            tgt_language = Language('English')
        else:
            src_language = Language('English')
            tgt_language = Language('French')

        pkl = {'src_language': src_language, 'tgt_language': tgt_language, 'pairs': pairs}
        pickle.dump(pkl, open(cache_path, 'wb'))
        print('cache stored...')

        return src_language, tgt_language, pairs

    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

    def filter_pair(self, p):
        return len(p[0].split(' ')) < self.max_length and len(p[1].split(' ')) < self.max_length \
               and p[1].startswith(SpecialToken.eng_prefixes)

    def prepare_data(self):
        src_language, tgt_language, pairs = self.read_dataset()
        print('read {} sentence pairs'.format(len(pairs)))
        pairs = self.filter_pairs(pairs)
        print('trimmed to {} sentence pairs'.format(len(pairs)))
        print('counting words...')
        for pair in pairs:
            src_language.add_sentence(pair[0])
            tgt_language.add_sentence(pair[1])
        print('words counting done')
        print(src_language.name, src_language.n_words)
        print(tgt_language.name, tgt_language.n_words)
        return src_language, tgt_language, pairs

    def prepare_data_loader(self, batch_size, num_workers):
        src_language, tgt_language, pairs = self.prepare_data()
        print(random.choice(pairs))

        pairs = [tensor_from_pair(src_language, tgt_language, pair) for pair in pairs]
        pairs, masks = add_padding_pairs(pairs, self.max_length)
        dtst = Src2Tgt(pairs, masks)
        pairs = data.DataLoader(dtst, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True)
        return src_language, tgt_language, pairs


class Src2Tgt(data.Dataset):
    def __init__(self, pairs, masks):
        super().__init__()
        assert np.shape(pairs) == np.shape(masks)
        self.fra2eng_pairs = pairs
        self.masks = masks

    def __len__(self):
        return len(self.fra2eng_pairs)

    def __getitem__(self, idx):
        return torch.tensor(self.fra2eng_pairs[idx], dtype=torch.long), torch.tensor(self.masks[idx], dtype=torch.float)
