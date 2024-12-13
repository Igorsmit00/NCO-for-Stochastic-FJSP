# Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods used for CP-SAT

We adapted the code from the original work by:

Robbert Reijnen, Kjell van Straaten, Zaharah Bukhsh, and Yingqian Zhang (2023) [Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods](https://arxiv.org/abs/2308.12794). arXiv preprint arXiv:2308.12794 

See the original repository [here](https://github.com/ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances).


## Running CP-SAT
To run the CP-SAT solver, you should create a TOML file with the appropriate structure, similar to the examples in the `configs` folder. Then, example commands are:

### Deterministic
```commandline
python run_cp_sat.py --config_file=configs/cp_sat_SD3_10x5+mix_det_1_1.toml
```
### Stochastic
```commandline
python run_cp_sat.py --config_file=configs/cp_sat_SD3_10x5+mix_stoch_VaR_1_25.toml
```

## Evaluating CP-SAT solutions
Once you finish running the CP-SAT solver, you can evaluate the solutions using the `run_realized_schedule.py` and `run_realized_schedules_folder.py` files. Example commands are:

### Deterministic solution
```commandline
python run_realized_schedules_folder.py --solution_directory="results/cp_sat/or_tools_3600/fjsp/SD3/10x5+mix" --instance_directory="fjsp/stoch/random/SD3/10x5+mix"
```

### Stochastic solution
```commandline
python run_realized_schedules_folder.py --solution_directory="results/cp_sat/or_tools_3600/stoch_25_VaR_95/fjsp/stoch/random/SD3/10x5+mix"  --instance_directory="fjsp/stoch/random/SD3/10x5+mix" --num_opt_realizations=25
```