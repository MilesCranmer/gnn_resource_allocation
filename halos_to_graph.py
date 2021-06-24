"""Convert a halo catalog into a graph"""
import torch
import sys
from torch.functional import F
from torch import optim, nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch import nn, optim
from data import get_dataloader, DATA_DIR
import pickle as pkl

run_num = int(sys.argv[1])

dataloader = get_dataloader(
    [run_num], cols=["x", "y", "z", "M14", "redshift"], subsample=1e-2
)

assert DATA_DIR is not None

graph = dataloader.dataset[0]
pkl.dump(
    graph,
    open(
        DATA_DIR + f"graph_obj_large_{run_num}.pkl",
        "wb",
    ),
)
