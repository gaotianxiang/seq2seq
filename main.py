import argparse
from utils import Params
import os
from director import Director


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--model_dir', '--md', default='experiments/base_model', type=str)
    parser.add_argument('--mode', '--m', default='train', type=str)
    parser.add_argument('--resume', '--r', action='store_true')
    args = parser.parse_args()

    hps_path = os.path.join(args.model_dir, 'config.json')
    if not os.path.exists(hps_path):
        raise FileNotFoundError('there is no config json file')
    hps = Params(hps_path)
    args.__dict__.update(hps.dict)
    return args


def main():
    args = get_parameters()
    director = Director(args)
    if args.mode == 'train':
        director.train()
    elif args.mode == 'eval':
        director.eval()
    elif args.mode == 'sample':
        director.sample()


if __name__ == '__main__':
    main()
