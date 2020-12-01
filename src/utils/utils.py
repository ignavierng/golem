from glob import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import tensorflow as tf


def get_init_path(outputs_dir='output/'):
    """Get the path of weighted matrix for initialization, by finding
    the output_dir with the latest datetime. It is assumed here that
    all output_dir contained in outputs_dir are formatted according
    to the datetime.

    Args:
        outputs_dir (str): Directory that containts a list of output_dir
            formatted according to datetime. Default: 'output/'.

    Returns:
        str: Path of weighted matrix for initialization.
    """
    return '{}/B_est.npy'.format(sorted(glob('{}/*'.format(outputs_dir)))[-2])


def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def count_accuracy(B_bin_true, B_bin_est, check_input=False):
    """Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1}, 
            where -1 indicates undirected edge in CPDAG.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        pred_size: prediction positive.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    if check_input:
        if (B_bin_est == -1).any():  # CPDAG
            if not ((B_bin_est == 0) | (B_bin_est == 1) | (B_bin_est == -1)).all():
                raise ValueError("B_bin_est should take value in {0, 1, -1}.")
            if ((B_bin_est == -1) & (B_bin_est.T == -1)).any():
                raise ValueError("Undirected edge should only appear once.")
        else:  # dag
            if not ((B_bin_est == 0) | (B_bin_est == 1)).all():
                raise ValueError("B_bin_est should take value in {0, 1}.")
            if not is_dag(B_bin_est):
                raise ValueError("B_bin_est should be a DAG.")
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_bin_est == -1)
    pred = np.flatnonzero(B_bin_est == 1)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'pred_size': pred_size}


def plot_solution(B_true, B_est, B_processed, save_name=None):
    """Checkpointing after the training ends.

    Args:
        B_true (numpy.ndarray): [d, d] weighted matrix of ground truth.
        B_est (numpy.ndarray): [d, d] estimated weighted matrix.
        B_processed (numpy.ndarray): [d, d] post-processed weighted matrix.
        save_name (str or None): Filename to solve the plot. Set to None
            to disable. Default: None.
    """
    fig, axes = plt.subplots(figsize=(10, 3), ncols=3)

    # Plot ground truth
    im = axes[0].imshow(B_true, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[0].set_title("Ground truth", fontsize=13)
    axes[0].tick_params(labelsize=13)

    # Plot estimated solution
    im = axes[1].imshow(B_est, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[1].set_title("Estimated solution", fontsize=13)
    axes[1].set_yticklabels([])    # Remove yticks
    axes[1].tick_params(labelsize=13)

    # Plot post-processed solution
    im = axes[2].imshow(B_processed, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[2].set_title("Post-processed solution", fontsize=13)
    axes[2].set_yticklabels([])    # Remove yticks
    axes[2].tick_params(labelsize=13)

    # Adjust space between subplots
    fig.subplots_adjust(wspace=0.1)

    # Colorbar (with abit of hard-coding)
    im_ratio = 3 / 10
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.05*im_ratio, pad=0.035)
    cbar.ax.tick_params(labelsize=13)
    # plt.show()

    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')
