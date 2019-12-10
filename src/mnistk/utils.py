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


def plot_error_splits(axis, y_true, y_pred, i):
    var_inv = [np.std(y_pred[y_true == j]) for j in range(10)]
    mx, mn = max(var_inv), min(var_inv)
    var_inv = [0.05 + 0.95 * (x - mn) / (mx - mn) for x in var_inv]
    for j in range(10):
        yj = y_pred[y_true == j]
        xj = [j] * len(yj)
        axis.scatter(yj, xj, s=var_inv[j], alpha=var_inv[j])
    axis.set_yticks(list(range(10)))
    axis.set_xlabel("Negative LogLikelihood")
    axis.set_ylabel("True Value")
    axis.set_title("Spread of Prediction = {} wrt Truth".format(i))


# TODO: What kind of wrong predictions are there? <11-12-19> #
def plot_hist_split(axis, y_true, y_pred, i):
    pass


def plot_roc(axis, y_true, y_pred, i):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    score = auc(fpr, tpr)
    axis.plot(fpr, tpr, color="#00FF00FF", label="AUC = {:.6f}".format(score))
    axis.set_ylabel("True Positive Rate")
    axis.set_xlabel("False Positive Rate")
    axis.set_title("ROC for Predicting {}".format(i))
    axis.legend()
    return score


def plot_images(epoch, img_tensors, pred_tensors, directory, writer=None):
    def clear_ticks(axis):
        axis.set_xticks([])
        axis.set_yticks([])

    def img(axis, z):
        clear_ticks(axis)
        z2, _, _ = nrm(z)
        axis.imshow(z2, cmap="gray_r", vmin=z2.min(), vmax=z2.max())

    def bprop(axis, z, n, **kwargs):
        clear_ticks(axis)
        axis.imshow(z, cmap="RdBu_r")
        axis.set_title("pred = {}".format(n))

    def nrm(z):
        z2 = z.detach().cpu().numpy()[0, 0, :, :]
        return z2, z2.min(), z2.max()

    with PdfPages(os.path.join(directory, "rf-{}.pdf".format(epoch))) as pdf:
        for i in range(10):
            fig = plt.figure(figsize=(10, 6))
            gsp = fig.add_gridspec(nrows=3, ncols=5)
            axs = list(fig.add_subplot(gsp[0, i]) for i in range(5)) + list(
                fig.add_subplot(gsp[2, i]) for i in range(5)
            )
            iax = fig.add_subplot(gsp[1, 2])
            img(iax, img_tensors[i])

            data = []
            for j in range(10):
                if img_tensors[i].grad is not None:
                    img_tensors[i].grad = None
                pred_tensors[i][0, j].backward(retain_graph=True)
                data.append(nrm(img_tensors[i].grad))

            arrs, mins, maxs = zip(*data)
            MIN, MAX = min(mins), max(maxs)
            for j in range(10):
                bprop(axs[j], z=arrs[j], n=j, vmin=MIN, vmax=MAX)
            fig.suptitle("Receptive field for {}".format(i))
            fig.subplots_adjust(wspace=0, hspace=0.5, right=0.9)
            if writer is not None:
                writer.add_figure(
                    tag="predictions/{}/recfield".format(i),
                    figure=fig,
                    global_step=epoch,
                    close=False,
                )

            pdf.savefig(fig, dpi=300)
            plt.close()
        d = pdf.infodict()
        d["Title"] = "Gradients at Epoch {}".format(epoch)


def plot_predictions(epoch, loss, y_true, y_pred, directory, writer=None):
    aucs = []
    accu = []
    preds = np.argmax(y_pred, axis=1)
    with PdfPages(os.path.join(directory, "test-{}.pdf".format(epoch))) as pdf:
        for i in range(10):
            fig = plt.figure(figsize=(9, 6))
            gsp = fig.add_gridspec(2, 3)
            ax = [fig.add_subplot(gsp[:, 0]), fig.add_subplot(gsp[:, 1:])]
            fig.suptitle("Network Predictions for {}".format(i))

            pred_check = preds[y_true == i]
            accu.append(np.sum(pred_check == i) / len(pred_check))

            yt = np.int32(y_true == i)
            yp = y_pred[:, i]
            aucs.append(plot_roc(ax[0], yt, yp, i))
            plot_error_splits(ax[1], y_true, yp, i)

            fig.subplots_adjust(wspace=0.5, hspace=0.5)
            pdf.savefig(fig, dpi=300)
            if writer is not None:
                writer.add_figure(
                    tag="predictions/{}/spread".format(i),
                    figure=fig,
                    global_step=epoch,
                    close=False,
                )
                writer.add_scalar(
                    "predictions/{}/auc".format(i), aucs[-1], global_step=epoch
                )
                writer.add_scalar(
                    "predictions/{}/accuracy".format(i), accu[-1], global_step=epoch
                )

            plt.close()
        d = pdf.infodict()
        d["Title"] = "Predictions at Epoch {}".format(epoch)
    if writer is not None:
        writer.add_scalar(tag="overall_test/loss", scalar_value=loss, global_step=epoch)
        writer.add_scalar(
            tag="overall_test/auc", scalar_value=np.mean(aucs), global_step=epoch
        )
        writer.add_scalar(
            tag="overall_test/accuracy", scalar_value=np.mean(accu), global_step=epoch
        )
    return aucs, accu


def save_predictions(y_pred, directory, run_name, epoch):
    predfile = os.path.join(directory, "predictions.h5")
    with h5py.File(predfile, "a") as f:  # rwa, create if exist
        name = "{}/{}".format(run_name, epoch)
        if f.get(name) is not None:
            ds = f.get(name)
            ds = y_pred
        else:
            f.create_dataset(name, data=y_pred)


def save_props(state, directory):
    with open(os.path.join(directory, "properties.json"), "w") as f:
        json.dump(state, f)
