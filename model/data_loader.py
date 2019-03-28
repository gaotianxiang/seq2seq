import random

from utils import tensor_from_pair
from torch.utils import data as data
from build_dataset import prepare_data


class Fra2Eng(data.Dataset):
    def __init__(self, pairs):
        super().__init__()
        self.fra2eng_pairs = pairs

    def __len__(self):
        return len(self.fra2eng_pairs)

    def __getitem__(self, idx):
        return self.fra2eng_pairs[idx]


def fetch_data_loader():
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', reverse=True)
    print(random.choice(pairs))

    pairs = [tensor_from_pair(input_lang, output_lang, pair) for pair in pairs]
    pairs = Fra2Eng(pairs)
    pairs = data.DataLoader(pairs, batch_size=1, shuffle=True, num_workers=0)
    return input_lang, output_lang, pairs
