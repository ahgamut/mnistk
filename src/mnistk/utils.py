# -*- coding: utf-8 -*-
"""
    mnistk.utils
    ~~~~~~~~~~~~~~

    Utility functions for plotting, saving etc.

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, auc
import numpy as np
import h5py
import os
import json

try:
    import torchviz
except Exception as e:
    print("No torchviz, can't visualize network")


def plot_structure(net, input_shapes, directory, fmt="svg", writer=None):
    try:
        g = torchviz.make_dot(net, input_shapes, render_depth=1)
        g.format = fmt
        g.render(filename="network", directory=directory, view=False, cleanup=True)
    except Exception as e:
        print(e)
        print("Unable to Visualize Network!!")


def plot_losses(train_losses, test_losses, directory):
    losses = np.array(train_losses)
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(1, len(losses) + 1), losses, "r.-", label="train")
    ax.plot(list(test_losses.keys()), list(test_losses.values()), "bo--", label="test")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_xticks(np.arange(1, len(losses) + 1))
    ax.legend()
    fig.savefig(os.path.join(directory, "loss.pdf"), dpi=300)
    plt.close()


def plot_hist_splits(axis, y_true, y_pred):
    x2 = y_pred[y_true == 1]
    x2h, x2b = np.histogram(x2, bins=20)
    x2w = (x2b[-1] - x2b[0]) / (len(x2b) - 1)
    x2b = x2b[:-1]

    x1 = y_pred[y_true == 0]
    x1h, x1b = np.histogram(x1, bins=x2b)
    x1w = (x1b[-1] - x1b[0]) / (len(x1b) - 1)
    x1b = x1b[:-1]

    axis.bar(x1b, x1h, align="edge", color="#88880088", label="truth=0", width=x1w)
    axis.bar(x2b, x2h, align="edge", color="#00888888", label="truth=1", width=x2w)
    axis.set_xlabel("Predicted Value")
    axis.set_ylabel("Frequency")
    axis.set_title("Distribution of Predictions")
    axis.legend()
    return (x1h, x1b), (x2h, x2b)


def plot_roc(axis, y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    score = auc(fpr, tpr)
    axis.plot(fpr, tpr, color="#00FF00FF", label="AUC = {:.6f}".format(score))
    axis.set_ylabel("True Positive Rate")
    axis.set_xlabel("False Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend()
    return score


def plot_images(ax1, img, ax2, grad):
    ax1.imshow(img, cmap="gray_r")
    ax1.set_title("Sample Image")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(grad, cmap="RdBu_r")

    nrm = Normalize(grad.min(), grad.max())
    mab = plt.cm.ScalarMappable(cmap="RdBu_r", norm=nrm)
    mab.set_array([])
    cb1 = ax2.figure.colorbar(
        mab,
        cax=None,
        ax=ax2,
        orientation="vertical",
        fraction=0.1,
        ticks=np.linspace(grad.min(), grad.max(), 5),
        format="%.4f",
        alpha=0.7,
    )
    ax2.set_title("RF Gradient")
    ax2.set_xticks([])
    ax2.set_yticks([])


def save_predictions(epoch, y_true, y_pred, bprops, directory, writer=None):
    aucs = []
    accu = np.sum(np.int32(np.argmax(y_pred, axis=1) == y_true)) / len(y_true)
    sample_grad = np.zeros((2, 1, 28, 28), np.float32)
    with PdfPages(os.path.join(directory, "test-{}.pdf".format(epoch))) as pdf:
        for i, sample, grad in bprops:
            fig, axs = plt.subplots(2, 2)
            ax = axs.ravel()
            fig.suptitle("Network Predictions for {}".format(i))
            yt = np.int32(y_true == i)
            yp = y_pred[:, i]
            aucs.append(plot_roc(ax[0], yt, yp))
            plot_hist_splits(ax[1], yt, yp)
            plot_images(ax[2], sample, ax[3], grad)

            fig.subplots_adjust(wspace=0.5, hspace=0.5)
            pdf.savefig(fig, dpi=300)
            if writer is not None:
                sample_grad[0, 0] = sample
                sample_grad[1, 0] = grad
                writer.add_figure(
                    tag="predictions/{}/overall".format(i),
                    figure=fig,
                    global_step=epoch,
                    close=False,
                )
                writer.add_scalar(
                    "predictions/{}/auc".format(i), aucs[-1], global_step=epoch
                )

            plt.close()
        d = pdf.infodict()
        d["Title"] = "Predictions at Epoch {}".format(epoch)

    with h5py.File(os.path.join(directory, "test-{}.h5".format(epoch)), "w") as f:
        g1 = f.create_group("roc")
        g1.create_dataset("truth", data=y_true)
        g1.create_dataset("preds", data=y_pred)

        g2 = f.create_group("samples")
        for i, sample, grad in bprops:
            g2.create_dataset("sample_{}".format(i), data=sample)
            g2.create_dataset("grad_{}".format(i), data=grad)

    return np.mean(aucs), accu


def save_props(state, directory):
    with open(os.path.join(directory, "properties.json"), "w") as f:
        json.dump(state, f)
