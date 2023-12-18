import argparse
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomNodeSplit, Compose, AddLaplacianEigenvectorPE

from experiments import run_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kl', type=int, default=9)
    parser.add_argument('--km', type=int, default=1)
    parser.add_argument('--kh', type=int, default=2)
    parser.add_argument('--t_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=3)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--att_dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.000005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    print(args)

    T = Compose([RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2), AddLaplacianEigenvectorPE(k=32)])
    dataset = Planetoid(root='/home/zq2/data', name='Cora', transform=T)
    data = dataset[0]

    data.x = torch.cat([data.x, data.laplacian_eigenvector_pe], dim=1)
    data.num_features = data.x.size(1)

    run_experiments(args.kl, args.km, args.kh, args.t_layers, args.num_heads, args.dropout, args.att_dropout,
                    args.lr, args.weight_decay, args.batch_size, args.epochs, args.runs, args.eval_steps,
                    data, len(data.y.unique()), args.cuda)

    print('============================================')
    print(args)


if __name__ == "__main__":
    main()
