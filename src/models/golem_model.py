import logging

import tensorflow as tf


class GolemModel:
    """Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, lambda_1, lambda_2, equal_variances=True,
                 seed=1, B_init=None):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            lambda_1 (float): Coefficient of L1 penalty.
            lambda_2 (float): Coefficient of DAG penalty.
            equal_variances (bool): Whether to assume equal noise variances
                for likelibood objective. Default: True.
            seed (int): Random seed. Default: 1.
            B_init (numpy.ndarray or None): [d, d] weighted matrix for
                initialization. Set to None to disable. Default: None.
        """
        self.n = n
        self.d = d
        self.seed = seed
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.equal_variances = equal_variances
        self.B_init = B_init

        self._build()
        self._init_session()

    def _init_session(self):
        """Initialize tensorflow session."""
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True
            )
        ))

    def _build(self):
        """Build tensorflow graph."""
        tf.compat.v1.reset_default_graph()

        # Placeholders and variables
        self.lr = tf.compat.v1.placeholder(tf.float32)
        self.X = tf.compat.v1.placeholder(tf.float32, shape=[self.n, self.d])
        self.B = tf.Variable(tf.zeros([self.d, self.d], tf.float32))
        if self.B_init is not None:
            self.B = tf.Variable(tf.convert_to_tensor(self.B_init, tf.float32))
        else:
            self.B = tf.Variable(tf.zeros([self.d, self.d], tf.float32))
        self.B = self._preprocess(self.B)

        # Likelihood, penalty terms and score
        self.likelihood = self._compute_likelihood()
        self.L1_penalty = self._compute_L1_penalty()
        self.h = self._compute_h()
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h

        # Optimizer
        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.score)
        self._logger.debug("Finished building tensorflow graph.")

    def _preprocess(self, B):
        """Set the diagonals of B to zero.

        Args:
            B (tf.Tensor): [d, d] weighted matrix.

        Returns:
            tf.Tensor: [d, d] weighted matrix.
        """
        return tf.linalg.set_diag(B, tf.zeros(B.shape[0], dtype=tf.float32))

    def _compute_likelihood(self):
        """Compute (negative log) likelihood in the linear Gaussian case.

        Returns:
            tf.Tensor: Likelihood term (scalar-valued).
        """
        if self.equal_variances:    # Assuming equal noise variances
            return 0.5 * self.d * tf.math.log(
                tf.square(
                    tf.linalg.norm(self.X - self.X @ self.B)
                )
            ) - tf.linalg.slogdet(tf.eye(self.d) - self.B)[1]
        else:    # Assuming non-equal noise variances
            return 0.5 * tf.math.reduce_sum(
                tf.math.log(
                    tf.math.reduce_sum(
                        tf.square(self.X - self.X @ self.B), axis=0
                    )
                )
            ) - tf.linalg.slogdet(tf.eye(self.d) - self.B)[1]

    def _compute_L1_penalty(self):
        """Compute L1 penalty.

        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return tf.norm(self.B, ord=1)

    def _compute_h(self):
        """Compute DAG penalty.

        Returns:
            tf.Tensor: DAG penalty term (scalar-valued).
        """
        return tf.linalg.trace(tf.linalg.expm(self.B * self.B)) - self.d


if __name__ == '__main__':
    # GOLEM-EV
    model = GolemModel(n=1000, d=20, lambda_1=2e-2, lambda_2=5.0,
                       equal_variances=True, seed=1)

    print("model.B: {}".format(model.B))
    print("model.likelihood: {}".format(model.likelihood))
    print("model.L1_penalty: {}".format(model.L1_penalty))
    print("model.h: {}".format(model.h))
    print("model.score: {}".format(model.score))
    print("model.train_op: {}".format(model.train_op))
