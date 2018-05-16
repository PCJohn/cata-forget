# cata-forget

Methods to reduce catastrophic interference in neural networks

1. Elastic Weight Consolidation

5 MNIST permutation tasks with EWC+Dropout:
<img src="res/fc_mnist_ewc_smooth.png" width="400">

Same tasks without EWC (only SGD+Dropout):
<img src="res/fc_mnist_sgd_dropout_smooth.png" width="400">

Saturation behavior

With EWC:
![Alt text](res/sat_mnist_ewc_smooth.png?raw=50x50)

Without EWC (SGD+dropout):
![Alt text](res/sat_sgd_dropout_smooth.png?=50x50)

Selectively forgetting (here, tasks 0,2,3):
![Alt text](res/sel_forget_023_smooth.png?=50x50)
