# Selectively Forgetting Tasks with Elastic Weight Consolidation

Elastic Weight Consolidation (https://arxiv.org/abs/1612.00796) allows a parameterized model to sequentially learn tasks with independent data. This project: (1) reproduces the results of the EWC paper (2) studies the saturation behavior of models (performance as more tasks are learned) and (3) implements a simple modification in the algorithm to learn tasks and then selectively forget some to free capacity.

The full report for the project can be seen here: <link to report>

To recreate the permuted MNIST experiments, run `mnist_permute_exp.py`. This will produce the following results:

1. Standard Elastic Weight Consolidation (reproducing the paper results)

5 permuted MNIST tasks with only SGD+Dropout:
<img src="res/fc_mnist_sgd_dropout_smooth.png" width="400">

Same tasks with EWC:
<img src="res/fc_mnist_ewc_smooth.png" width="400">

Saturation behavior

With EWC:
![Alt text](res/sat_mnist_ewc_smooth.png?raw=30x30)

Without EWC (SGD+dropout):
![Alt text](res/sat_sgd_dropout_smooth.png?=30x30)

Selectively forgetting (here, tasks 0,2,3):
![Alt text](res/sel_forget_023_smooth.png?=30x30)
