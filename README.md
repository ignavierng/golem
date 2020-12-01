# Structure Learning with GOLEM

This repository contains an implementation of the structure learning method described in ["On the Role of Sparsity and DAG Constraints for Learning Linear DAGs"](https://arxiv.org/abs/2006.10201). 

If you find it useful, please consider citing:
```bibtex
@inproceedings{Ng2020role,
  author = {Ng, Ignavier and Ghassami, AmirEmad and Zhang, Kun},
  title = {{On the Role of Sparsity and DAG Constraints for Learning Linear DAGs}},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020},
}
```

## TL;DR
We formulate a likelihood-based score function with soft sparsity and DAG constraints for learning linear DAGs, which guarantees learning a DAG equivalent to the ground truth DAG, under mild assumption. This leads to an unconstrained optimization problem that can be solved via  gradient-based optimization method.

## Requirements

Python 3.6+ is required. To install the requirements:
```setup
pip install -r requirements.txt
```

(**Optional**) For GPU support, install `tensorflow-gpu==1.15.0` (with CUDA and cuDNN), e.g., through conda:
```setup
conda install tensorflow-gpu==1.15.0
```

## Running GOLEM-EV
The hyperparameters for GOLEM-EV are:
- `equal_variances=True`
- `lambda_1=2e-2`
- `lambda_2=5.0`
- `num_iter=1e+5`
- `learning_rate=1e-3`.

To run GOLEM-EV:
```
# Ground truth: 20-node ER2 graph
# Data: Linear DAG model with Gaussian-NV noise
python src/main.py  --seed 1 \
                    --d 20 \
                    --graph_type ER \
                    --degree 4 \
                    --noise_type gaussian_nv \
                    --equal_variances \
                    --lambda_1 2e-2 \
                    --lambda_2 5.0 \
                    --checkpoint_iter 5000
```
Each run creates a directory based on current datetime to save the training outputs, e.g., `output/2020-12-01_12-11-50-562`.

## Running GOLEM-NV
The hyperparameters for GOLEM-NV are:
- `equal_variances=False`
- `lambda_1=2e-3`
- `lambda_2=5.0`
- `num_iter=1e+5`
- `learning_rate=1e-3`.

The optimization problem of GOLEM-NV is susceptible to local solutions, so we have to initialize it with the solution returned by GOLEM-EV. The hyperparameters of GOLEM-EV are described in the previous section.

There are two ways to initializate the optimization problem:

**(1)** Set `init` to `True`. By default, the code will load the estimated solution of the **latest** experiment (based on datetime) in the `output` directory.\
(Please make sure the latest experiment indeed corresponds to GOLEM-EV with same dataset configurations.)
```
# Ground truth: 20-node ER2 graph
# Data: Linear DAG model with Gaussian-NV noise
python src/main.py  --seed 1 \
                    --d 20 \
                    --graph_type ER \
                    --degree 4 \
                    --noise_type gaussian_nv \
                    --non_equal_variances \
                    --init \
                    --lambda_1 2e-3 \
                    --lambda_2 5.0 \
                    --checkpoint_iter 5000
```

**(2)** Set `init` to `True` and manually set `init_path` to the path of estimated solution (`.npy` file) by GOLEM-EV.
```
# Ground truth: 20-node ER2 graph
# Data: Linear DAG model with Gaussian-NV noise
python src/main.py  --seed 1 \
                    --d 20 \
                    --graph_type ER \
                    --degree 4 \
                    --noise_type gaussian_nv \
                    --non_equal_variances \
                    --init \
                    --init_path <PATH_TO_NUMPY_MATRIX> \
                    --lambda_1 2e-3 \
                    --lambda_2 5.0 \
                    --checkpoint_iter 5000
```
Each run creates a directory based on current datetime to save the training outputs, e.g., `output/2020-12-01_12-11-50-562`.

## Examples
- See [golem.py](src/golem.py#L50) for a minimal usage of GOLEM.
- See [GOLEM-EV.ipynb](examples/GOLEM-EV.ipynb) and [GOLEM-NV.ipynb](examples/GOLEM-NV.ipynb) for a demo.
- Example of solution returned by GOLEM-EV:
    - Ground truth: 20-node ER2 graph
    - Data: Linear DAG model with Gaussian-EV noise.

    <img width="600" alt="example" src="https://user-images.githubusercontent.com/20400992/100785208-5662bd80-33de-11eb-853f-350992076ae5.png"/>

## Acknowledgments
- The code to generate the synthetic data and compute the metrics (e.g., SHD, TPR) is based on [NOTEARS](https://github.com/xunzheng/notears/blob/master/notears/utils.py).
- We are grateful to the authors of the baseline methods for releasing their code.