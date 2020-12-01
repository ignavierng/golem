import numpy as np

from utils.utils import count_accuracy, plot_solution, is_dag


def threshold_till_dag(B):
    """Remove the edges with smallest absolute weight until a DAG is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.
    """
    if is_dag(B):
        return B, 0

    B = np.copy(B)
    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(B[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(B):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        B[j, i] = 0
        dag_thres = abs(weight)

    return B, dag_thres


def postprocess(B, graph_thres=0.3):
    """Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.
        graph_thres (float): Threshold for weighted matrix. Default: 0.3.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
    """
    B = np.copy(B)
    B[np.abs(B) <= graph_thres] = 0    # Thresholding
    B, _ = threshold_till_dag(B)

    return B


def checkpoint_after_training(output_dir, X, B_true, B_init,
                              B_est, B_processed, print_func):
    """Checkpointing after the training ends.

    Args:
        output_dir (str): Output directory to save training outputs.
        X (numpy.ndarray): [n, d] data matrix.
        B_true (numpy.ndarray): [d, d] weighted matrix of ground truth.
        B_init (numpy.ndarray or None): [d, d] weighted matrix for
            initialization. Set to None to disable. Default: None.
        B_est (numpy.ndarray): [d, d] estimated weighted matrix.
        B_processed (numpy.ndarray): [d, d] post-processed weighted matrix.
        print_func (function): Printing function.
    """
    # Visualization
    plot_solution(B_true, B_est, B_processed,
                  save_name='{}/plot_solution.jpg'.format(output_dir))
    print_func("Finished plotting estimated graph (without post-processing).")

    results = count_accuracy(B_true != 0, B_processed != 0)
    print_func("Results (after post-processing): {}.".format(results))

    # Save training outputs
    np.save('{}/X.npy'.format(output_dir), X)
    np.save('{}/B_true.npy'.format(output_dir), B_true)
    np.save('{}/B_est.npy'.format(output_dir), B_est)
    np.save('{}/B_processed.npy'.format(output_dir), B_processed)
    if B_init is not None:
        np.save('{}/B_init.npy'.format(output_dir), B_init)
    print_func("Finished saving training outputs at {}.".format(output_dir))
