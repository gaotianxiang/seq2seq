from model.data_loader import fetch_data_loader
import argparse


def t_data_loader():
    args = argparse.Namespace()
    args.max_length = 10
    args.batch_size = 5

    _, _, dtld = fetch_data_loader(args)
    for p, m in dtld:
        print(p)
        print(m)
        print(p.size())
        print(m.size())
        p1 = p[:, 0, :]
        m1 = m[:, 0]
        print(p1)
        print(m1)
        print(type(p1))
        print(type(m1))
        print(p1.size())
        print(m1.size())
        break


if __name__ == '__main__':
    t_data_loader()
