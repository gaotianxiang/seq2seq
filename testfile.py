import sys
from main import unicode2ascii, normalize_string


def test_normalize_string(s):
    print('original: ')
    print('\t\t\t', s)
    s = normalize_string(s)
    print(type(s))
    print(s)


if __name__ == '__main__':
    test_normalize_string('    gao tianXian xiang.@    #!!!     ')
