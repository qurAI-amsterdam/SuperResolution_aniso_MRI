import yaml
import argparse
import numpy as np
import torch
from datetime import datetime


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def kl_divergence(mu, logvar):
    """
    Compute KL-Divergence.
    Important: mu and logvar are supposed to have shape [batch, -1]

    Another important note: *** we are not averaging over the batch dimension ***
    If necessary (has huge effect on training) this should be done after calling this function

    :param mu:
    :param logvar:
    :return:
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(reconstruction, image, mu, logvar, loss_func):
    b, c, H, W = image.size()
    mu = mu.contiguous().view((b, -1))
    logvar = logvar.contiguous().view((b, -1))
    recon_loss = loss_func(reconstruction, image)
    kl_loss = kl_divergence(mu, logvar)  # 0.5 * torch.sum(torch.exp(logvar) + mean**2 - 1. - logvar)
    return recon_loss, kl_loss


def load_settings(fname):
    with open(fname, 'r') as fp:
        args = yaml.load(fp, Loader=yaml.FullLoader)
    return args


def save_settings(args, fname):
    with open(fname, 'w') as fp:
        yaml.dump(vars(args), fp)


def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp, Loader=yaml.FullLoader))
    return args


def saveExperimentSettings(args, fname):
    if isinstance(args, dict):
        with open(fname, 'w') as fp:
            yaml.dump(args, fp)
    else:
        with open(fname, 'w') as fp:
            yaml.dump(vars(args), fp)


def one_hot_encoding(labels, classes=[0, 1, 2, 3]):
    """

    :param labels: have shape [y, x] OR [z, y, x] or [2, z, y, x]  type np.int
    :return: binarized labels [4, y, x] OR [z, n_classes, y, x] OR [z, 2xn_classes, y, x] WHERE 0:4=ES and 4:8=ED
    """
    if labels.shape[0] == 2 and labels.ndim == 4:
        # shape is [2, w, h, d]
        # we are dealing with a combined ES/ED label array
        array_es = np.stack([(labels[0] == cls_idx).astype(np.int) for cls_idx in classes], axis=0)
        array_ed = np.stack([(labels[1] == cls_idx).astype(np.int) for cls_idx in classes], axis=0)
        binary_array = np.concatenate([array_es, array_ed], axis=0)
    else:
        if labels.ndim == 4:
            raise ValueError("ERROR - binarize acdc labels - shape of array is 4 but first dim != 2 (ES/ED)")
        elif labels.ndim == 2 or labels.ndim == 3:
            # shape is [x, y] OR [z, y, x]
            binary_array = np.stack([(labels == cls_idx).astype(np.int) for cls_idx in classes], axis=0)
            if labels.ndim == 3:
                binary_array = binary_array.transpose((1, 0, 2, 3))

        else:
            raise ValueError("ERROR - binarize labels acdc - Rank {} of array not supported".format(labels.ndim))

    return binary_array


def compute_entropy(pred_probs, dim=1, eps=1e-7):
    """

    :param pred_probs: shape [z, 4, x, y]
    :param eps:
    :return:
    """
    if isinstance(pred_probs, torch.Tensor):
        # convert to numpy array
        pred_probs = pred_probs.detach().cpu().numpy()
    entropy = (-pred_probs * np.log2(pred_probs + eps)).sum(axis=dim)
    entropy = np.nan_to_num(entropy)
    # we sometimes encounter tiny negative values
    umap_max = 2.
    umap_min = np.min(entropy)
    entropy = (entropy - umap_min) / (umap_max - umap_min)
    return entropy


def compute_entropy_pytorch(p, dim=1, keepdim=False, eps=1e-7):
    p = p + eps
    if keepdim:
        # return -torch.where(p > 0, p * p.log2(), p.new([0.0])).sum(dim=dim, keepdim=True)
        return -(p * p.log2()).sum(dim=dim, keepdim=True)
    else:
        # return -torch.where(p > 0, p * p.log2(), p.new([0.0])).sum(dim=dim, keepdim=True).squeeze()
        return -(p * p.log2()).sum(dim=dim, keepdim=True).squeeze()


def generate_exper_id(exper_id=None):
    now = datetime.now()
    if exper_id is None:
        return now.strftime("%m%d%H%M")
    else:
        return exper_id + "_" + now.strftime("%m%d%H%M")
