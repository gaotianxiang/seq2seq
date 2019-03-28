from torch.utils import data as data


class Fra2Eng(data.Dataset):
    def __init__(self, pairs):
        super().__init__()
        self.fra2eng_pairs = pairs

    def __len__(self):
        return len(self.fra2eng_pairs)

    def __getitem__(self, idx):
        return self.fra2eng_pairs[idx]
