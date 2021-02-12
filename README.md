# Framework for Training GNN (using PyG)

## GNN

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
### data

For computation using SPMM operations, we use the concept of NodeBlock

NodeBlock is a list of Adjacency matrices (SparseTensor) including:

- num_layers
- layers_adj
- layers_nodes
- layers_aux (for future use)


NodeBlock has .to() method for transfering to CUDA/CPU

#### Sampler

TBF: not complete yet

---

### dataset
Dataset is loaded using PyG

    from gnn import dataset
    dataset('cora')

PyG datasets are loaded into pyg.dataset.data consist of:
- edge_index
- train_mask
- val_mask
- test_mask
- x: features
- y: labels

---

### layers


---

### models

---

### Training

The base class is for selecting the model/optimizer/loss.

3 main function train/validition and inferece

Two modes are supported now Full and Batch

In Full with Base as parent class:

The graph is loaded once and any preprocessing (normalization) is applied on the graph.
One node block is created and features and labels are transfered to the `self.device`.

- run()

Pre-Process 'graph' self.data
Create a shared 'nodeblock' self.nbs from 'graph' self.data

load features and labels
self.x
self.y

- train()
- validation()
- inference()

In Batch with Full as parent class:

validation and 



## Notebooks

## Tests

## Examples
