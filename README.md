# GPPool

The source code of **Graph Partial Pooling: A Universal Approach for Scaling Up GNNs**


## Training
To reproduce the results of GPPool on the Cora dataset, please run following commands.
```
Conda activate your_conda_env_name
python test.py 
```
If you need to apply GPPool to other datasets, simply modify the dataset section in test.py and use the hyperparameters suggested in the paper for training.


## Citing

Please cite our work if you find it is useful for you:
```
@article{LowpassFilteringLargescale,
  title = {Beyond Low-Pass Filtering on Large-Scale Graphs via Adaptive Filtering Graph Neural Networks},
  author = {Zhang, Qi and Li, Jinghua and Sun, Yanfeng and Wang, Shaofan and Gao, Junbin and Yin, Baocai},
  date = {2024-01},
  journaltitle = {Neural Networks},
  volume = {169},
  pages = {1--10}
}
```
