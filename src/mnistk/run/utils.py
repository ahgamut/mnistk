# -*- coding: utf-8 -*-
"""
    mnistk.utils
    ~~~~~~~~~~~~~~

    Utility functions for plotting, saving etc.

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import h5py
import os
import json

try:
    import torchrec
except Exception as e:
    print("No torchrec, can't visualize network")


def get_recorder(net, input_shapes):
    try:
        rec = torchrec.record(net, input_shapes, name=net.__class__.__name__)
        return rec
    except Exception as e:
        print(e)
        print("Unable to Visualize Network!!")


def plot_structure(rec, directory, fmt="svg"):
    g = torchrec.make_dot(rec, render_depth=1)
    g.format = fmt
    g.render(filename="network", directory=directory, view=False, cleanup=True)


def layerop_count(rec):
    opcount = 0
    layercount = 0
    for fn in rec.node_set:
        if "Backward" in fn.__class__.__name__ and fn is rec.nodes[fn].fn:
            # checking fn because there are dummy ops in the recorder
            opcount += 1
        if fn is rec.nodes[fn].fn and hasattr(fn, "children"):
            has_children = len(list(fn.children())) != 0
            layercount += 0 if has_children else 1
    return layercount, opcount


# TODO: A better measure of memory <21-12-19> #
def approx_mem_usage(rec):
    mem = 0
    for n in rec.nodes.values():
        n.duplicates = 0
    for fn in rec.node_set:
        rec.nodes[fn].duplicates += 1
    ans = list([rec.nodes[x] for x in rec.node_set if x is not None])
    for n in ans:
        if hasattr(n.fn, "size"):
            mem += n.fn.numel() / n.duplicates
    return mem


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
    js = np.array([x for x in range(10)])
    width = 0.25
    f_neg = [np.sum((y_pred == j) & (y_true == i)) for j in js]
    f_pos = [np.sum((y_pred == i) & (y_true == j)) for j in js]
    f_neg[i] = 0
    f_pos[i] = 0
    fnb = axis.bar(
        x=js - width / 2,
        height=f_neg,
        width=width,
        edgecolor="#0000FFFF",
        facecolor="#0000FF88",
        label="False Negative",
    )
    fpb = axis.bar(
        x=js + width / 2,
        height=f_pos,
        width=width,
        edgecolor="#FF0000FF",
        facecolor="#FF000088",
        label="False Positive",
    )

    def text_above(rs, ls):
        for r, l in zip(rs, ls):
            if l == 0:
                continue
            axis.text(
                r.get_x() + 0.5 * r.get_width(),
                r.get_height() * 1.01,
                str(l),
                ha="center",
                va="bottom",
            )

    text_above(fnb, f_neg)
    text_above(fpb, f_pos)
    axis.set_xticks(js)
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Frequency")
    axis.set_title("Distribution of WRONG Predictions")
    axis.legend()


def plot_roc(axis, y_true, y_pred, i):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    score = auc(fpr, tpr)
    axis.plot(fpr, tpr, color="#00FF00FF", label="AUC = {:.6f}".format(score))
    axis.set_ylabel("True Positive Rate")
    axis.set_xlabel("False Positive Rate")
    axis.set_title("ROC for Predicting {}".format(i))
    axis.legend()
    return score


def plot_images(epoch, img_tensors, pred_tensors, directory):
    def clear_ticks(axis):
        axis.set_xticks([])
        axis.set_yticks([])

    def img(axis, z):
        clear_ticks(axis)
        z2, _, _ = nrm(z)
        return axis.imshow(z2, cmap="gray_r", vmin=z2.min(), vmax=z2.max())

    def bprop(axis, z, n, **kwargs):
        clear_ticks(axis)
        axis.set_title("pred = {}".format(n))
        return axis.imshow(z, cmap="RdBu_r", **kwargs)

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
            mappables = []
            for j in range(10):
                mappables.append(bprop(axs[j], z=arrs[j], n=j, vmin=MIN, vmax=MAX))
            fig.suptitle("Receptive field for {}".format(i))
            fig.subplots_adjust(
                wspace=0, hspace=0.25, left=0.0, right=0.8, bottom=0.05, top=0.9
            )
            cax = fig.add_axes([0.85, 0.05, 0.03, 0.85])  # left bottom width height
            cb = fig.colorbar(mappable=mappables[0], cax=cax, ax=axs)
            pdf.savefig(fig, dpi=300)
            plt.close()
        d = pdf.infodict()
        d["Title"] = "Gradients at Epoch {}".format(epoch)


def get_classwise_preds(y_true, y_pred):
    aucs = []
    accus = []
    pred_val = np.argmax(y_pred, axis=1)
    yt = np.zeros(len(y_true), np.float32)
    for i in range(10):
        pred_i = pred_val[y_true == i]
        accu = np.sum(pred_i == i) / len(pred_i)
        yt *= 0
        yt += y_true == i
        auc = roc_auc_score(yt, y_pred[:, i])

        accus.append(accu)
        aucs.append(auc)
    return aucs, accus


def plot_predictions(epoch, loss, y_true, y_pred, directory):
    aucs = []
    accus = []
    preds = np.argmax(y_pred, axis=1)
    with PdfPages(os.path.join(directory, "test-{}.pdf".format(epoch))) as pdf:
        for i in range(10):
            fig = plt.figure(figsize=(9, 6))
            gsp = fig.add_gridspec(6, 3)
            ax = [fig.add_subplot(gsp[0:2, :]), fig.add_subplot(gsp[3:, :])]
            fig.suptitle("Network Predictions for {}".format(i))

            pred_check = preds[y_true == i]
            accus.append(np.sum(pred_check == i) / len(pred_check))

            yt = np.int32(y_true == i)
            yp = y_pred[:, i]
            aucs.append(plot_roc(ax[0], yt, yp, i))
            plot_error_splits(ax[1], y_true, preds, i)

            pdf.savefig(fig, dpi=300)

            plt.close()
        d = pdf.infodict()
        d["Title"] = "Predictions at Epoch {}".format(epoch)
    return aucs, accus


def save_predictions(y_pred, directory, run_name, epoch):
    predfile = os.path.join(directory, "predictions.h5")
    with h5py.File(predfile, "a") as f:  # rwa, create if exist
        name = "{}/{}".format(run_name, epoch)
        if f.get(name) is not None:
            ds = f.get(name)
            ds = y_pred
        else:
            f.create_dataset(name, data=y_pred)


def save_props(state, directory, filename="properties.json"):
    with open(os.path.join(directory, filename), "w") as f:
        json.dump(state, f)
