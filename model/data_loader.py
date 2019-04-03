import random
import torch
import numpy as np
import argparse

from utils import tensor_from_pair
from torch.utils import data as data
from build_dataset import prepare_data


class Fra2Eng(data.Dataset):
    def __init__(self, pairs, masks):
        super().__init__()
        assert np.shape(pairs) == np.shape(masks)
        self.fra2eng_pairs = pairs
        self.masks = masks

    def __len__(self):
        return len(self.fra2eng_pairs)

    def __getitem__(self, idx):
        return torch.tensor(self.fra2eng_pairs[idx], dtype=torch.long), torch.tensor(self.masks[idx], dtype=torch.float)


def add_padding(sentence: list, max_length):
    """according to max_length add padding to sentence

    Args:
        sentence: list of words eg. ['I', 'am', 'tianxiang', '.']
        max_length: integer maximum length of sentence

    Returns:
        a padded sentence eg. ['I', 'am', 'tianxiang', '.', 'PAD'] if max_length = 5
        a mask (list) which has 1's at normal word entries and 0's at PAD entries e.g. [1, 1, 1, 1, 0]
    """
    current_length = len(sentence)
    mask = np.zeros(max_length)
    sentence = sentence + [0] * (max_length - current_length)
    mask[:current_length] = 1
    return sentence, mask


def add_padding_pairs(pairs, args):
    """add paddings to list of language pairs

    Args:
        pairs: list of lists [ [torch.tensor, torch.tensor], [torch.tensor, torch.tensor], ...]
            each torch.tensor contains list of long int, which is the index of words
        args: contain all hyper parameters

    Returns:
        padded pairs
        corresponding masks
    """
    max_length = args.max_length
    padded_pairs = []
    masks = []
    for lang1, lang2 in pairs:
        lang1, mask_lang1 = add_padding(lang1, max_length)
        lang2, mask_lang2 = add_padding(lang2, max_length)
        padded_pairs.append([lang1, lang2])
        masks.append([mask_lang1, mask_lang2])
    return padded_pairs, masks


def fetch_data_loader(args):
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', args, reverse=True)
    print(random.choice(pairs))

    pairs = [tensor_from_pair(input_lang, output_lang, pair) for pair in pairs]
    pairs, masks = add_padding_pairs(pairs, args)
    dtst = Fra2Eng(pairs, masks)
    pairs = data.DataLoader(dtst, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    return input_lang, output_lang, pairs
