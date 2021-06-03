# Source code for On Provable Benefits of Depth in Training Graph Convolutional Networks

The experiments are seperated into two parts, 
- The code for Open Graph Benchmark can be find in folder `./OGB_exps/`
- The code for synthetic, Cora, Citeseer datasets can be find in folder `./small_graph_exps/`

## Reproduce results on OGB datasets


This code relies on latest version of PyG and OGB, for best experience instead of using pre-built, use the latest code from github and install them using:

    git clone git@github.com:snap-stanford/ogb.git
    python setup.py develop


do the same for the following repos:

    git@github.com:rusty1s/pytorch_geometric.git
    git@github.com:rusty1s/pytorch_sparse.git
    git@github.com:rusty1s/pytorch_scatter.git
    git@github.com:rusty1s/pytorch_cluster.git
    git@github.com:rusty1s/pytorch_spline_conv.git

---

### Scripts
To run different models use the files in script folder. example:

    python ogb-arxiv.py <output_folder> <num_repeat> <model> <num_layers>

---
### Data

For computation using SPMM operations, we use the concept of NodeBlock

NodeBlock is a list of Adjacency matrices (SparseTensor) including:

- num_layers
- layers_adj
- layers_nodes
- layers_aux (for future use)


NodeBlock has .to() method for transfering to CUDA/CPU

---

### Dataset
Dataset is loaded using PyG

    from gnn import dataset
    dataset('cora')

Specify the enviroment variable `GNN_DATASET_DIR` to select where the data is stored!


---

### Layers

---

### Models

---

### Training

The base class is for selecting the model/optimizer/loss.

- run():
- train()
- validation()
- inference()


## Examples
