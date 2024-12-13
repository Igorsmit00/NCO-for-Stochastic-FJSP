# Extension of Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning using the Scenario Processing Module

We adapted the code from the original work by:

Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Siew Tan, Chi Xu. Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning. 34th Conference on Neural Information Processing Systems (NeurIPS), 2020. [\[PDF\]](https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf)

See the original repository [here](https://github.com/zcaicaros/L2D).


To generate random instances from the deterministic instances, run the following command:
```commandline
python3 generate_random_instances.py
```

To train a model, use the following command:
```commandline
python3 python PPO_jssp_multiInstances.py
```

To evaluate a trained model, use the following command:
```commandline
python3 test_learned.py
```

Note that to train a deterministic model, the original repo can be used. A model trained using this repo can also be evaluated in our repo.