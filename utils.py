import torch

from build_dataset import Language, EOS_token, SOS_token


class RunningAverage:
    def __init__(self):
        self.counts = 0
        self.total_sum = 0
        self.avg = 0

    def reset(self):
        self.counts = 0
        self.total_sum = 0

    def update(self, val):
        self.total_sum += val
        self.counts += 1
        self.avg = self.total_sum / self.counts


def indexes_from_sentence(lang: Language, sentence: str):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang: Language, sentence: str):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensor_from_pair(input_lang: Language, output_lang: Language, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    output_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, output_tensor
