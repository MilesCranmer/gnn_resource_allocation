#!/bin/bash
conda env create --name $1 -f environment.yml && \
    export TORCH="1.7.0+cu110" && \
    conda activate $1 && \
    pip install --upgrade --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}.html && \
    pip install --upgrade --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}.html && \
    pip install --upgrade --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}.html && \
    pip install --upgrade --no-cache-dir torch-geometric
