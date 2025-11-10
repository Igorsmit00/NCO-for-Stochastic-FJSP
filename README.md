# Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems
This repo contains the code of our paper **[Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems](https://arxiv.org/abs/2412.14052)**, accepted at the *39th Annual AAAI Conference on Artificial Intelligence (AAAI-25)*.


The repo consists of 3 sub-repos. These sub-repos are used as follows:
## FJSP-DRL
The FJSP-DRL folder contains the main code of our project. Here, we apply the scenario processing module to the dual attention network to create SPM-DAN.
This folder contains the code for DRL training and evaluation, as well as the code for the scenario processing module and the dual attention network. 
In addition, the code for the dispatching rules is also included in this folder.

## Job_Shop_Scheduling_Benchmark_Environments_and_Instances
The Job_Shop_Scheduling_Benchmark_Environments_and_Instances folder contains the code for the CP-SAT solver.
It has both the deterministic CP-SAT formulation and our stochastic CP-stoch formulation. Both the code for creating schedules using CP-SAT and evaluating those schedules is included.

## L2D
The L2D folder contains the code for our preliminary experiment in which we apply the scenario processing module to the L2D network.

Within each of these folders, there is a README file that explains the code in more detail.

## Reference
To cite our work, please refer to the BibTeX below.
```bibtex
@inproceedings{smit2025neural,
  title={Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems},
  author={Smit, Igor G. and Wu, Yaoxin and Troubil, Pavel and Zhang, Yingqian and Nuijten, Wim P.M.},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={25},
  pages={26678--26687},
  year={2025},
  month={4},
  doi={10.1609/aaai.v39i25.34870},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/34870}
}
```
