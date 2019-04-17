import unicodedata
import re
import numpy as np


class SpecialToken:
    SOS_token = 1
    EOS_token = 2
    Pad_token = 0

    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )


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


def indexes_from_sentence(lang: Language, sentence: str):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang: Language, sentence: str):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(SpecialToken.EOS_token)
    return indexes


def tensor_from_pair(input_lang: Language, output_lang: Language, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    output_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, output_tensor


def unicode2ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'M')


def normalize_string(s: str):
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s


def add_padding(sentence: list, max_length):
    """according to max_length add padding to sentence

    Args:
        sentence: list of indices of words
        max_length: integer maximum length of sentence

    Returns:
        a padded sentence eg. [3, 24, 45, 2, 0] if max_length = 5
        a mask (list) which has 1's at normal word entries and 0's at PAD entries e.g. [1, 1, 1, 1, 0]
    """
    current_length = len(sentence)
    mask = np.zeros(max_length)
    sentence = sentence + [0] * (max_length - current_length)
    mask[:current_length] = 1
    return sentence, mask


def add_padding_pairs(pairs, max_length):
    """add paddings to list of language pairs

    Args:
        pairs: list of lists [ [torch.tensor, torch.tensor], [torch.tensor, torch.tensor], ...]
            each torch.tensor contains list of long int, which is the index of words
        args: contain all hyper parameters

    Returns:
        padded pairs
        corresponding masks
    """
    padded_pairs = []
    masks = []
    for lang1, lang2 in pairs:
        lang1, mask_lang1 = add_padding(lang1, max_length)
        lang2, mask_lang2 = add_padding(lang2, max_length)
        padded_pairs.append([lang1, lang2])
        masks.append([mask_lang1, mask_lang2])
    return padded_pairs, masks
