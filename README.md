# Unsupervised Resource Allocation with Graph Neural Networks

![](https://github.com/MilesCranmer/gnn_resource_allocation/blob/master/schematic.svg)

Check out the paper [here](https://arxiv.org/abs/2106.09761).

PyTorch code for the forward model of our algorithm can be found in our repository [here](https://github.com/MilesCranmer/gnn_resource_allocation/blob/master/model.py)

[Data](https://app.globus.org/file-manager?origin_id=75a68b36-a6c0-11eb-92d8-6b08dd67ff48&origin_path=%2F)

Requirements:

- torch
- torch-geometric
- einops
- scikit-learn
- numpy
- [Pylians](https://github.com/franciscovillaescusa/Pylians3)
- My astropy branch:
```
pip install git+https://github.com/MilesCranmer/astropy
```
(it has some of the Cosmology calculations vectorized)

If you are using `conda`, and have CUDA version 11.0 and cuDNN version 8.0, you can create a duplicate of our env, using:
```bash
./create_env.sh gnn_allocation
```
which will create a new environment called `gnn_allocation`. This uses PyTorch 1.7.1, though it is likely to work for other versions if you decide to modify `create_env.sh` and `environment.yml`.

You will then need to install Pylians manually. To do this, check out the
[repo](https://github.com/franciscovillaescusa/Pylians3),
and follow these instructions: https://pylians3.readthedocs.io/en/master/linux.html#option-1
