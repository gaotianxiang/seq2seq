import unicodedata
import re
import pickle
import os
import numpy as np

SOS_token = 1
EOS_token = 2
Pad_token = 0


class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'PAD', 1: 'SOS', 2: 'EOS'}
        self.n_words = 3

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
    cache_path = './data/fra-eng/fra-eng.preprocess'
    if os.path.exists(cache_path):
        print('cache hit, read from cache...')
        pkl = pickle.load(open(cache_path, 'rb'))
        return pkl['input_lang'], pkl['output_lang'], pkl['pairs']
    print('cache miss...')
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

    pkl = {'input_lang': input_lang, 'output_lang': output_lang, 'pairs': pairs}
    pickle.dump(pkl, open(cache_path, 'wb'))
    print('cache stored...')

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


def filter_pair(p, args):
    return len(p[0].split(' ')) < args.max_length and len(p[1].split(' ')) < args.max_length and p[1].startswith(
        eng_prefixes)


def filter_pairs(pairs, args):
    return [pair for pair in pairs if filter_pair(pair, args)]


def prepare_data(lang1, lang2, args, reverse=False):
    input_lang, output_lang, pairs = read_languages(lang1, lang2, reverse)
    print('read {} sentence pairs'.format(len(pairs)))
    pairs = filter_pairs(pairs, args)
    print('trimmed to {} sentence pairs'.format(len(pairs)))
    print('counting words...')
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print('words counting done')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
