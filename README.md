# Unsupervised Resource Allocation with Graph Neural Networks

![](https://github.com/MilesCranmer/gnn_resource_allocation/blob/master/schematic.svg)

Check out the paper [here](https://arxiv.org/abs/2106.09761).

PyTorch code for the forward model of our algorithm can be found in this repository in the file `model.py`. To train the model, execute `train.py`.

Data required to train this model can be found [here](https://app.globus.org/file-manager?origin_id=75a68b36-a6c0-11eb-92d8-6b08dd67ff48&origin_path=%2F)

Requirements for our codebase can be found in `environment.yml`. Note that one needs to use the following custom astropy:
```
pip install git+https://github.com/MilesCranmer/astropy
```
(it has some of the Cosmology calculations vectorized).

If you are using `conda`, and have CUDA version 11.0 and cuDNN version 8.0, you can create a duplicate of our env, using:
```bash
./create_env.sh gnn_allocation
```
which will create a new environment called `gnn_allocation`. This uses PyTorch 1.7.1, though it is likely to work for other versions if you decide to modify `create_env.sh` and `environment.yml`. You can also use an implementation without CUDA using the `environment_nocuda.yml` file.
