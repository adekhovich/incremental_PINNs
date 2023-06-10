# iPINNs: Incremental learning for Physics-informed neural networks

This repository contains the official implementation of iPINNs (https://arxiv.org/abs/2304.04854).

## Introduction
Physics-informed neural networks (PINNs) have recently become a powerful tool for solving partial differential equations (PDEs). However, finding a set of neural network parameters that lead to fulfilling a PDE can be challenging and non-unique due to the complexity of the loss landscape that needs to be traversed. Although a variety of multi-task learning and transfer learning approaches have been proposed to overcome these issues, there is no incremental training procedure for PINNs that can effectively mitigate such training challenges. We propose incremental PINNs (iPINNs) that can learn multiple tasks (equations) sequentially without additional parameters for new tasks and improve performance for every equation in the sequence. Our approach learns multiple PDEs starting from the simplest one by creating its own subnetwork for each PDE and allowing each subnetwork to overlap with previously learned subnetworks. We demonstrate that previous subnetworks are a good initialization for a new equation if PDEs share similarities. We also show that iPINNs achieve lower prediction error than regular PINNs for two different scenarios: (1) learning a family of equations (e.g., 1-D convection PDE); and (2) learning PDEs resulting from a combination of processes (e.g., 1-D reaction-diffusion PDE). The ability to learn all problems with a single network together with learning more complex PDEs with better generalization than regular PINNs will open new avenues in this field.

## Installation

* Clone this github repository using:

      git clone https://github.com/adekhovich/incremental_PINNs.git

* Install requirements using:

      pip install -r requirements.txt
      
## Train the model

Run the code with:

      python3 src/main.py
      

## Examples

* To replicate our experiments on 1-D convection equation, use the following command:
```
python3 src/main.py --optimizer_name Adam --lr 1e-2 --weight_decay 0 --num_epochs 20000\
                    --activation sin --num_tasks 5 --alpha_fc 0.95 --system convection\
                    --a 0.0 --b 1.0 --N_f 1000 --nt 100 --xgrid 256 --seed 0
```

* To replicate our experiments on the sequense of reaction, diffusion and reaction-diffusion equations, use the following command:
```
python3 src/main.py --optimizer_name Adam --lr 1e-2 --weight_decay 0 --num_epochs 20000\
                    --activation sin --num_tasks 3 --alpha_fc 0.95 --system rd\
                    --u0_str gauss --a 0.0 --b 1.0 --N_f 1000 --nt 100 --xgrid 256 --seed 0
```

* If you want to use our code to train a regular PINN, for example on 1-D convection with `β = 10`, use the following command:
```
python3 src/main.py --optimizer_name Adam --lr 1e-2 --weight_decay 0 --num_epochs 20000\
                    --activation sin --num_tasks 1 --system convection --beta 10\
                    --a 0.0 --b 1.0 --N_f 1000 --nt 100 --xgrid 256 --seed 0
```
      
     
## Acknowledgements

Our code implementation is based on the following GitHub project: https://github.com/a1k12/characterizing-pinns-failure-modes, which is the official implementation of the work ["Characterizing possible failure modes in physics-informed neural networks"](https://arxiv.org/abs/2109.01050), *Advances in Neural Information Processing Systems* **34** (2021) by Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, Michael W. Mahoney.
      
      
## Citation

If you use our code in your research, please cite our work:
```
@article{dekhovich2023ipinns,
  title={iPINNs: Incremental learning for Physics-informed neural networks},
  author={Dekhovich, Aleksandr and Sluiter, Marcel HF and Tax, David MJ and Bessa, Miguel A},
  journal={arXiv preprint arXiv:2304.04854},
  year={2023}
}
```     
      
      
