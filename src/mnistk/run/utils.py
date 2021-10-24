# -*- coding: utf-8 -*-
"""
    mnistk.utils
    ~~~~~~~~~~~~~~

    Utility functions for plotting, saving etc.

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import h5py
import os
import json
import base64

try:
    import torchrecorder
except Exception as e:
    print("No torchrecorder, can't visualize network")


class NDArrayEncoder(json.JSONEncoder):
    """JSON encoder for numpy ndarrays"""

    # https://stackoverflow.com/questions/27909658/json-encoder-and-decoder-for-complex-numpy-arrays
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "shape": obj.shape,
                "data": base64.b64encode(obj.data).decode("utf-8"),
                "dtype": str(obj.dtype),
            }
        return json.JSONEncoder.default(self, obj)


def ndarray_hook(obj):
    if not isinstance(obj, dict):
        return obj
    keys = list(obj.keys())
    if "shape" in keys and "data" in keys:
        return np.reshape(
            np.frombuffer(base64.b64decode(obj["data"]), dtype=obj["dtype"]),
            obj["shape"],
        )
    return obj


class NDArrayDecoder(json.JSONDecoder):
    """JSON Decoder for numpy ndarrays"""

    def __init__(self, **kwargs):
        super(NDArrayDecoder, self).__init__(object_hook=ndarray_hook, **kwargs)


def get_recorder(net, input_shapes):
    try:
        rec = torchrecorder.record(net, input_shapes, name=net.__class__.__name__)
        return rec
    except Exception as e:
        print(e)
        print("Unable to Visualize Network!!")


def plot_structure(rec, directory, fmt="svg"):
    g = torchrecorder.make_dot(rec, render_depth=1)
    g.format = fmt
    g.render(filename="network", directory=directory, view=False, cleanup=True)


def layerop_count(rec):
    opcount = 0
    layercount = 0
    for node in set(rec.nodes.values()):
        if type(node).__name__ == "OpNode":
            opcount += 1
        if hasattr(node.fn, "children"):
            has_children = len(list(node.fn.children())) != 0
            layercount += 0 if has_children else 1
    return layercount, opcount


# TODO: A better measure of memory <21-12-19> #
def approx_mem_usage(rec):
    mem = 0
    for n in set(rec.nodes.values()):
        if hasattr(n.fn, "size") and "Tensor" in type(n).__name__:
            mem += n.fn.numel()
    return mem


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


def get_classwise_preds(y_true, y_pred):
    aucs = np.zeros(10, dtype=np.float32)
    accus = np.zeros(10, dtype=np.float32)
    pred_val = np.argmax(y_pred, axis=1)
    yt = np.zeros(len(y_true), np.float32)
    for i in range(10):
        pred_i = pred_val[y_true == i]
        accu = np.sum(pred_i == i) / len(pred_i)
        yt *= 0
        yt += y_true == i
        auc = roc_auc_score(yt, y_pred[:, i])

        accus[i] = accu
        aucs[i] = auc
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


def save_props(record_dict, directory, filename="properties.json"):
    record_dict["train loss"] = np.array(record_dict["train loss"], dtype=np.float32)
    for k in record_dict["test accuracy"].keys():
        record_dict["test accuracy"][k] = np.array(
            record_dict["test accuracy"][k], dtype=np.float32
        )
        record_dict["test AUC"][k] = np.array(
            record_dict["test AUC"][k], dtype=np.float32
        )
    with open(os.path.join(directory, filename), "w") as f:
        json.dump(record_dict, f, cls=NDArrayEncoder)
