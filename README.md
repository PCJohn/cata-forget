# cata-forget

Methods to reduce catastrophic interference in neural networks

1. Elastic Weight Consolidation

5 MNIST permutation tasks with EWC+Dropout:
![Alt text](res/fc_mnist_ewc.png?raw=50x50)

Same tasks without EWC (only SGD+Dropout):
![Alt text](res/fc_mnist_sgd_dropout.png?raw=50x50)

Behavior after saturation

With EWC:
![Alt text](res/fc_mnist_ewc_35.png?raw=50x50)

Without EWC (SGD+dropout):
![Alt text](res/sat_sgd_dropout_35.png?=50x50)
