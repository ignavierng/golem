import logging

import numpy as np
import tensorflow as tf

from utils.dir import create_dir


class GolemTrainer:
    """Set up the trainer to solve the unconstrained optimization problem of GOLEM."""
    _logger = logging.getLogger(__name__)

    def __init__(self, learning_rate=1e-3):
        """Initialize self.

        Args:
            learning_rate (float): Learning rate of Adam optimizer.
                Default: 1e-3.
        """
        self.learning_rate = learning_rate

    def train(self, model, X, num_iter, checkpoint_iter=None, output_dir=None):
        """Training and checkpointing.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.
            num_iter (int): Number of iterations for training.
            checkpoint_iter (int): Number of iterations between each checkpoint.
                Set to None to disable. Default: None.
            output_dir (str): Output directory to save training outputs. Default: None.

        Returns:
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        model.sess.run(tf.compat.v1.global_variables_initializer())

        self._logger.info("Started training for {} iterations.".format(num_iter))
        for i in range(0, int(num_iter) + 1):
            if i == 0:    # Do not train here, only perform evaluation
                score, likelihood, h, B_est = self.eval_iter(model, X)
            else:    # Train
                score, likelihood, h, B_est = self.train_iter(model, X)

            if checkpoint_iter is not None and i % checkpoint_iter == 0:
                self.train_checkpoint(i, score, likelihood, h, B_est, output_dir)

        return B_est

    def eval_iter(self, model, X):
        """Evaluation for one iteration. Do not train here.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.

        Returns:
            float: value of score function.
            float: value of likelihood function.
            float: value of DAG penalty.
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        score, likelihood, h, B_est \
            = model.sess.run([model.score, model.likelihood, model.h, model.B],
                             feed_dict={model.X: X,
                                        model.lr: self.learning_rate})

        return score, likelihood, h, B_est

    def train_iter(self, model, X):
        """Training for one iteration.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.

        Returns:
            float: value of score function.
            float: value of likelihood function.
            float: value of DAG penalty.
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        _, score, likelihood, h, B_est \
            = model.sess.run([model.train_op, model.score, model.likelihood, model.h, model.B],
                             feed_dict={model.X: X,
                                        model.lr: self.learning_rate})

        return score, likelihood, h, B_est

    def train_checkpoint(self, i, score, likelihood, h, B_est, output_dir):
        """Log and save intermediate results/outputs.

        Args:
            i (int): i-th iteration of training.
            score (float): value of score function.
            likelihood (float): value of likelihood function.
            h (float): value of DAG penalty.
            B_est (numpy.ndarray): [d, d] estimated weighted matrix.
            output_dir (str): Output directory to save training outputs.
        """
        self._logger.info(
            "[Iter {}] score {:.3E}, likelihood {:.3E}, h {:.3E}".format(
                i, score, likelihood, h
            )
        )

        if output_dir is not None:
           # Save the weighted matrix (without post-processing)
            create_dir('{}/checkpoints'.format(output_dir))
            np.save('{}/checkpoints/B_iteration_{}.npy'.format(output_dir, i), B_est)
