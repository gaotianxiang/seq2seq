import argparse
import wget

parse = argparse.ArgumentParser()
parse.add_argument('--url', type=str)
parse.add_argument('--out_dir', type=str, default='./')

args = parse.parse_args()

url = args.url
out_dir = args.out_dir

print('- downloading from {}'.format(url))
file = wget.download(url, out_dir)
print('- downloaded')
