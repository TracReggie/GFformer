import argparse
import torch
from torch_geometric.transforms import RandomNodeSplit, Compose, ToUndirected

from experiments import run_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kl', type=int, default=7)
    parser.add_argument('--km', type=int, default=1)
    parser.add_argument('--kh', type=int, default=1)
    parser.add_argument('--t_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=1)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--att_dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.000005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--cuda', type=int, default=2)
    args = parser.parse_args()
    print(args)

    T = Compose([RandomNodeSplit('train_rest', num_val=0.25, num_test=0.25), ToUndirected()])
    data = torch.load('data/Amazon2m.pt')
    data = T(data)

    run_experiments(
        args.kl, args.km, args.kh, args.t_layers, args.num_heads, args.dropout, args.att_dropout, args.lr,
        args.weight_decay, args.batch_size, args.epochs, args.runs, args.eval_steps, data, len(data.y.unique()),
        args.cuda)

    print('============================================')
    print(args)


if __name__ == "__main__":
    main()
