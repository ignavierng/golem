import os

from models import GolemModel
from trainers import GolemTrainer


# For logging of tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def golem(X, lambda_1, lambda_2, equal_variances=True,
          num_iter=1e+5, learning_rate=1e-3, seed=1,
          checkpoint_iter=None, output_dir=None, B_init=None):
    """Solve the unconstrained optimization problem of GOLEM, which involves
        GolemModel and GolemTrainer.

    Args:
        X (numpy.ndarray): [n, d] data matrix.
        lambda_1 (float): Coefficient of L1 penalty.
        lambda_2 (float): Coefficient of DAG penalty.
        equal_variances (bool): Whether to assume equal noise variances
            for likelibood objective. Default: True.
        num_iter (int): Number of iterations for training.
        learning_rate (float): Learning rate of Adam optimizer. Default: 1e-3.
        seed (int): Random seed. Default: 1.
        checkpoint_iter (int): Number of iterations between each checkpoint.
            Set to None to disable. Default: None.
        output_dir (str): Output directory to save training outputs.
        B_init (numpy.ndarray or None): [d, d] weighted matrix for initialization.
            Set to None to disable. Default: None.

    Returns:
        numpy.ndarray: [d, d] estimated weighted matrix.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """
    # Set up model
    n, d = X.shape
    model = GolemModel(n, d, lambda_1, lambda_2, equal_variances, seed, B_init)

    # Training
    trainer = GolemTrainer(learning_rate)
    B_est = trainer.train(model, X, num_iter, checkpoint_iter, output_dir)

    return B_est    # Not thresholded yet


if __name__ == '__main__':
    # Minimal code to run GOLEM.
    import logging

    from data_loader import SyntheticDataset
    from utils.train import postprocess
    from utils.utils import count_accuracy, set_seed

    # Setup for logging
    # Required for printing histories if checkpointing is activated
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s - %(name)s - %(message)s'
    )

    # Reproducibility
    set_seed(1)

    # Load dataset
    n, d = 1000, 20
    graph_type, degree = 'ER', 4    # ER2 graph
    B_scale = 1.0
    noise_type = 'gaussian_ev'
    dataset = SyntheticDataset(n, d, graph_type, degree,
                               noise_type, B_scale, seed=1)

    # GOLEM-EV
    B_est = golem(dataset.X, lambda_1=2e-2, lambda_2=5.0,
                  equal_variances=True, checkpoint_iter=5000)

    # Post-process estimated solution and compute results
    B_processed = postprocess(B_est, graph_thres=0.3)
    results = count_accuracy(dataset.B != 0, B_processed != 0)
    logging.info("Results (after post-processing): {}.".format(results))
