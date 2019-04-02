import torch

from build_dataset import Language, EOS_token, SOS_token
import json
import logging
from tqdm import tqdm


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


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def set_logger(log_path, terminal=True):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
        terminal: whether add console handler
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        if terminal:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)


def log(msg):
    logging.info(msg)
    tqdm.write(msg)


def indexes_from_sentence(lang: Language, sentence: str):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang: Language, sentence: str):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return indexes


def tensor_from_pair(input_lang: Language, output_lang: Language, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    output_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, output_tensor
