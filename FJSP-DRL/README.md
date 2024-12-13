# Extension of FJSP-DRL using the Scenario Processing Module

We adapted the code from the original work by:

“[Flexible Job Shop Scheduling via Dual Attention Network Based Reinforcement Learning](https://doi.org/10.1109/TNNLS.2023.3306421)”. *IEEE Transactions on Neural Networks and Learning Systems*, 2023.

See the original repository [here](https://github.com/wrqccc/FJSP-DRL).

## Introduction

- `data` saves the instance files including testing instances (in the subfolder `BenchData`, `SD1` and `SD2`) and validation instances (in the subfolder `data_train_vali`).
- `model` contains the implementation of the network.
- `test_results`saves the results solved by priority dispatching rules and DRL models.
- `train_log` saves the training log of models, including information of the reward and validation makespan.
- `common_utils.py` contains some useful functions (including the implementation of priority dispatching rules mentioned in the paper) .
- `data_utils.py` is used for data generation, reading and format conversion.
- `fjsp_env_same_op_nums.py` and `fjsp_env_various_op_nums.py` are implementations of fjsp environments, describing fjsp instances with the same number of operations and different number of operations, respectively. These are the environments from the original work.
- `fjsp_env_same_op_nums_torch.py` and `fjsp_env_various_op_nums_torch.py` are implementations of fjsp environments, describing fjsp instances with the same number of operations and different number of operations, respectively. These are the environments adapted to work with PyTorch to facilitate parallelization.
- `fjsp_env_same_op_nums_torch_no_feat.py` and `fjsp_env_various_op_nums_torch_no_feat.py` are implementations of fjsp environments, describing fjsp instances with the same number of operations and different number of operations, respectively. These are the environments adapted to work with PyTorch to facilitate parallelization, but without features. These are used to efficiently compute the reward values.
- `params.py` defines parameters settings.
- `test_heuristic_stoch.py` is used for solving the instances by priority dispatching rules.
- `test_trained_model_stochastic.py` is used for evaluating the models.
- `train.py` is used for training and `train_deterministic.py` is used for training the deterministic baseline DAN models.
- `generate_random_instances.py` and `run_generate_instances_often.py` are used for generating random instances from the deterministic instanes.

## Train

```commandline
python train.py
```

## Evaluate

```commandline
python test_trained_model_stochastic.py
```

Note that before training and evaluation, you need to generate the random instances. Due to the size of the instance files, we do not provide the instances in this repository. You can generate the instances using the `generate_random_instances.py` and `run_generate_instances_often.py` files.


## Example Commands
To train and evaluate different models, with different instances, the arguments from params.py can be modified. We provide some examples below.

### Train with the Scenario Processing Module

```commandline
python train.py  --SAA_attention="True" --variance="random" --n_j=10 --n_m=5 --stoch_obj="VaR" --minibatch_size=1024 --gradient_accumulation_steps=1 --SAA_attention_dim=32 --hidden_dim_actor=128 --hidden_dim_critic=64 --layer_fea_output_dim 32 8
```

### Train without the Scenario Processing Module but with stochastic rewards

```commandline
python train.py  --SAA_attention="False" --variance="random" --n_j=10 --n_m=5 --stoch_obj="VaR" --minibatch_size=1024 --gradient_accumulation_steps=1 --SAA_attention_dim=32 --hidden_dim_actor=64 --hidden_dim_critic=64 --layer_fea_output_dim 32 8
```

### Train without the Scenario Processing Module and with deterministic rewards

```commandline
python train_deterministic.py  --SAA_attention="False" --n_j=10 --n_m=5 --minibatch_size=1024 --gradient_accumulation_steps=1 --hidden_dim_actor=64 --hidden_dim_critic=64 --layer_fea_output_dim 32 8 --data_source=SD1
```

### Train a policy with a different number of state scenarios

```commandline
python test_trained_model_stochastic.py --model_source=SD3 --data_source=SD3 --test_data=10x5+mix --test_model=ran10x5+mix_stoch_attn_200_SAA_VaR95_32_8_128_64 --variance=random --num_input_realizations=200 --num_eval_realizations=1000 --SAA_attention=True --test_mode=False --deter_model=False --SAA_attention_dim=32 --hidden_dim_actor=128 --hidden_dim_critic=64 --layer_fea_output_dim 32 8
```


### Evaluate a policy with the Scenario Processing Module
#### Greedy
```commandline
python test_trained_model_stochastic.py --model_source=SD3 --data_source=SD3 --test_data=10x5+mix --test_model=ran10x5+mix_stoch_attn_SAA_VaR95_32_8_128_64 --variance=random --num_input_realizations=100 --num_eval_realizations=1000 --SAA_attention=True --test_mode=False --deter_model=False --SAA_attention_dim=32 --hidden_dim_actor=128 --hidden_dim_critic=64 --layer_fea_output_dim 32 8
```
#### Sample
```commandline
python test_trained_model_stochastic.py --model_source=SD3 --data_source=SD3 --test_data=10x5+mix --test_model=ran10x5+mix_stoch_attn_SAA_VaR95_32_8_128_64 --variance=random --num_input_realizations=100 --num_eval_realizations=1000 --SAA_attention=True --test_mode=True --deter_model=False --SAA_attention_dim=32 --hidden_dim_actor=128 --hidden_dim_critic=64 --layer_fea_output_dim 32 8
```

### Evaluate a policy without the Scenario Processing Module but trained with stochastic rewards
#### Greedy
```commandline
python test_trained_model_stochastic.py --model_source=SD3 --data_source=SD3 --test_data=10x5+mix --test_model=ran10x5+mix_stoch_SAA_VaR95_32_8_64_64 --variance=random --num_input_realizations=100 --num_eval_realizations=1000 --SAA_attention=False --test_mode=False --deter_model=False --SAA_attention_dim=32 --hidden_dim_actor=64 --hidden_dim_critic=64 --layer_fea_output_dim 32 8
```
#### Sample
```commandline
python test_trained_model_stochastic.py --model_source=SD3 --data_source=SD3 --test_data=10x5+mix --test_model=ran10x5+mix_stoch_SAA_VaR95_32_8_64_64 --variance=random --num_input_realizations=100 --num_eval_realizations=1000 --SAA_attention=False --test_mode=True --deter_model=False --SAA_attention_dim=32 --hidden_dim_actor=64 --hidden_dim_critic=64 --layer_fea_output_dim 32 8
```

### Evaluate a policy without the Scenario Processing Module and trained with deterministic rewards
#### Greedy
```commandline
python test_trained_model_stochastic.py --model_source=SD3 --data_source=SD3 --test_data=10x5+mix --test_model=10x5+mix --variance=random --num_input_realizations=100 --num_eval_realizations=1000 --SAA_attention=False --test_mode=False --deter_model=True --SAA_attention_dim=32 --hidden_dim_actor=64 --hidden_dim_critic=64 --layer_fea_output_dim 32 8
```
#### Sample
```commandline
python test_trained_model_stochastic.py --model_source=SD3 --data_source=SD3 --test_data=10x5+mix --test_model=10x5+mix --variance=random --num_input_realizations=100 --num_eval_realizations=1000 --SAA_attention=False --test_mode=True --deter_model=True --SAA_attention_dim=32 --hidden_dim_actor=64 --hidden_dim_critic=64 --layer_fea_output_dim 32 8
```

### Running Heuristics
#### FIFO example
```commandline
python test_heuristic_stoch.py --data_source=SD3 --test_data=10x5+mix --variance=random --num_eval_realizations=1000 --test_method FIFO