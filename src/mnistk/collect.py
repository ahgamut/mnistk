# -*- coding: utf-8 -*-
"""
    mnistk.collect
    ~~~~~~~~~~~~~~

    Collect data from JSONs in subfolders into a CSV

    :copyright: (c) 2020 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from os.path import join as osjoin, dirname, isfile
from os import remove
from torchvision.datasets import MNIST
import json
import glob
import pandas as pd
import numpy as np
import h5py


def column_names(sortable=False, static=False):
    static_cols = ["#layers", "#params", "#ops", "memory per pass"]
    score_cols = (
        ["time"]
        + ["accuracy"]
        + ["accuracy_{}".format(x) for x in range(10)]
        + ["AUC"]
        + ["AUC_{}".format(x) for x in range(10)]
    )
    cols = (
        ["name", "activation", "groupname", "formname", "run", "epoch", "loss"]
        + static_cols
        + score_cols
        + ["rank", "rank_gp", "rank_form", "rank_snap"]
    )
    if sortable:
        if static:
            return static_cols
        return score_cols
    return cols


def get_groupname(name):
    for x in ["Conv1dThenLinear", "Conv2dThenLinear", "Conv3dThenLinear"]:
        if x in name:
            return x
    for x in ["Basic", "ResNet", "Conv1d", "Conv2d", "Conv3d", "Linear"]:
        if x in name:
            return x
    return name


def get_records(jdict):
    records = []
    for epoch in jdict["test loss"].keys():
        record = {}
        record["epoch"] = int(epoch)
        record["formname"] = jdict["name"].split("_")[0]
        record["groupname"] = get_groupname(jdict["name"])
        for k, v in jdict.items():
            if k in ["test AUC", "test accuracy"]:
                modkey = k.replace("test ", "")
                record[modkey] = sum(jdict[k][epoch]) / 10
                for x in range(10):
                    record["{}_{}".format(modkey, x)] = jdict[k][epoch][x]
            elif k == "test loss":
                modkey = k.replace("test ", "")
                record[modkey] = jdict[k][epoch]
            elif k == "time per epoch":
                record["time"] = jdict[k] * int(epoch)
            elif k == "train loss":
                continue
            else:
                record[k] = jdict[k]
        for rnk in ["rank", "rank_gp", "rank_form", "rank_snap"]:
            record[rnk] = 0
        records.append(record)
    return records


def append_records(csv_dict, json_path, static_info):
    with open(json_path) as f:
        jdict = json.load(f)
        jdict.update(static_info)
        records = get_records(jdict)
        for record in records:
            for k, v in record.items():
                csv_dict[k].append(v)


def write_to_csv(result_dir):
    print("Creating summary.csv...")
    csv_path = osjoin(result_dir, "summary.csv")
    colnames = column_names()
    csv_dict = {col: [] for col in colnames}
    splist = glob.glob(osjoin(result_dir, "**/network.json"), recursive=True)
    for static_path in splist:
        with open(static_path, "r") as sf:
            static_info = json.load(sf)
            jplist = glob.glob(
                osjoin(dirname(static_path), "**/properties.json"), recursive=True
            )
            for json_path in jplist:
                append_records(csv_dict, json_path, static_info)

    df = pd.DataFrame(csv_dict)
    df["rank"] = df["accuracy"].rank(method="min", ascending=False)
    df["rank_gp"] = df.groupby("groupname")["accuracy"].rank("dense", ascending=False)
    df["rank_form"] = df.groupby("formname")["accuracy"].rank("dense", ascending=False)
    df["rank_snap"] = df.groupby(["run", "epoch"])["accuracy"].rank(
        "dense", ascending=False
    )
    df.to_csv(csv_path, header=True, index=False)


def split_predictions(pred_path):
    if isfile(pred_path):
        with h5py.File(pred_path, "r") as apf:
            for k1 in apf.keys():
                snap_pred = osjoin(
                    dirname(pred_path), "runs/{}/predictions.h5".format(k1)
                )
                with h5py.File(snap_pred, "w") as snapf:
                    for k2 in apf[k1].keys():
                        ds = np.array(apf[k1][k2])
                        dsp = np.argmax(ds, axis=1)
                        snapf.create_dataset("/{}/preds".format(k2), data=dsp)
                        snapf.create_dataset("/{}/scores".format(k2), data=ds)
        remove(pred_path)


def update_exam(main_dir, exam_arr):
    split_predictions(osjoin(main_dir, "predictions.h5"))
    predlist = glob.glob(osjoin(main_dir, "runs/**/predictions.h5"), recursive=True)
    for path in predlist:
        with h5py.File(path, "r") as pf:
            for grp in pf.values():
                preds = grp.get("preds")
                exam_arr[range(len(preds)), preds] += 1


def save_exam_scores(result_dir):
    print("Creating exam stats on test set...")
    exam_scores = np.zeros((10000, 10), dtype=np.float32)
    splist = glob.glob(osjoin(result_dir, "**/network.json"), recursive=True)
    for static_path in splist:
        update_exam(dirname(static_path), exam_scores)
    with h5py.File(osjoin(result_dir, "exam_stats.h5"), "w") as f:
        exam_percent = exam_scores / np.sum(exam_scores, axis=1, keepdims=True)
        exam_percent = exam_percent
        f.create_dataset("scores", data=exam_percent)
        data = MNIST("./data", train=False)
        true_vals = data.targets.numpy()
        f.create_dataset("truth", data=true_vals)


def save_rankings(result_dir):
    print("Saving Rankings for each data point...")
    main_df = pd.read_csv(osjoin(result_dir, "summary.csv"), header=0)
    # memory is a problem when main_df has many rows
    ranking_dfs = {
        x: pd.DataFrame() for x in ["rank", "rank_gp", "rank_form", "rank_snap"]
    }
    ranking_group = {
        "rank": "",
        "rank_gp": "groupname",
        "rank_form": "formname",
        "rank_snap": ["run", "epoch"],
    }
    asc_cols = column_names(sortable=True, static=True) + ["time"]
    desc_cols = column_names(sortable=True, static=False)[1:]
    for k, v in ranking_dfs.items():
        v["name"] = main_df["name"]
        v["run"] = main_df["run"]
        v["epoch"] = main_df["epoch"]
        v["out of"] = 0
        if ranking_group[k] == "":
            df2 = main_df
            v["out of"] = len(main_df)
        else:
            df2 = main_df.groupby(ranking_group[k])
            t = main_df[ranking_group[k]]
            if isinstance(ranking_group[k], str):
                v["out of"] = main_df[ranking_group[k]].apply(
                    lambda x: len(df2.get_group(x))
                )
            else:
                v["out of"] = main_df[ranking_group[k]].apply(
                    lambda x: len(df2.get_group(tuple(x))), axis=1
                )
        for asc_col in asc_cols:
            v[asc_col] = df2[asc_col].rank(method="dense", ascending=True)
        for desc_col in desc_cols:
            v[desc_col] = df2[desc_col].rank(method="dense", ascending=False)

    def get_ranks(name, run, epoch, cols):
        ans2 = pd.DataFrame(
            index=["global", "in_group", "in_form", "in_run"], columns=cols
        )

        def onerow(key):
            _df = ranking_dfs[key]
            _df2 = _df[
                (_df["name"] == name) & (_df["run"] == run) & (_df["epoch"] == epoch)
            ]
            return _df2[cols].iloc[0]

        ans2.loc["global"] = onerow("rank")
        ans2.loc["in_group"] = onerow("rank_gp")
        ans2.loc["in_form"] = onerow("rank_form")
        ans2.loc["in_run"] = onerow("rank_snap")
        return ans2.T.to_dict("split")

    def get_jdict(path):
        with open(path, "r") as jfile:
            _jdict = json.load(jfile)
        return dirname(path), _jdict

    splist = glob.glob(osjoin(result_dir, "**/network.json"), recursive=True)
    for static_path in splist:
        static_dir, sdict = get_jdict(static_path)
        jplist = glob.glob(
            osjoin(dirname(static_path), "**/properties.json"), recursive=True
        )
        with open(osjoin(static_dir, "rankings.json"), "w") as sf:
            stat_dict = get_ranks(sdict["name"], "t1", 1, asc_cols[:-1] + ["out of"])
            json.dump(stat_dict, sf)

        for json_path in jplist:
            json_dir, jdict = get_jdict(json_path)
            ans_dict = dict()
            for ep in jdict["test accuracy"].keys():
                epi = int(ep)
                ans_dict[ep] = get_ranks(
                    sdict["name"], jdict["run"], epi, ["time"] + desc_cols + ["out of"]
                )
            with open(osjoin(json_dir, "rankings.json"), "w") as f:
                json.dump(ans_dict, f)
