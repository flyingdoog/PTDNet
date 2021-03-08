## Tensorflow Version: GraphSage: Representation Learning on Large Graphs


### Enviroments
* Tensorflow 1.13
* Python 3.7

### Revise
* Remove Id_map
* Read Data from GCN form
* Unsupervised Learning. remove val etc.


### Supervised Version.
* support 'task_type=semi' and 'task_type=full'. For cora, with 'semi', we use 140 nodes for trainning (splitted by GCN paper). With 'full' we use all left nodes, apart from validation and testing, for trainning.
* currently, I do not tuning the hyper-parameters.
* Upload 'cora' and 'citeseer' datasets.
