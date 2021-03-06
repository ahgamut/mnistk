# -*- coding: utf-8 -*-
"""
    mnistk.collect
    ~~~~~~~~~~~~~~

    Collect data from JSONs in subfolders into a CSV

    :copyright: (c) 2020 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from os.path import join as osjoin, dirname, isfile, abspath
from os import remove
from torchvision.datasets import MNIST
import json
import glob
import pandas as pd
import numpy as np
import h5py
from mnistk import NDArrayDecoder, NDArrayEncoder
from mnistk.run.utils import save_props
import sqlalchemy as sa


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


def get_records(record_dict):
    records = []
    for epoch in record_dict["test loss"].keys():
        record = {}
        record["epoch"] = int(epoch)
        record["formname"] = record_dict["name"].split("_")[0]
        record["groupname"] = get_groupname(record_dict["name"])
        for k, v in record_dict.items():
            if k in ["test AUC", "test accuracy"]:
                modkey = k.replace("test ", "")
                record[modkey] = record_dict[k][epoch].mean().tolist()
                for x in range(10):
                    record["{}_{}".format(modkey, x)] = record_dict[k][epoch][
                        x
                    ].tolist()
            elif k == "test loss":
                modkey = k.replace("test ", "")
                record[modkey] = record_dict[k][epoch]
            elif k == "time per epoch":
                record["time"] = record_dict[k] * int(epoch)
            elif k == "train loss":
                continue
            else:
                record[k] = record_dict[k]
        for rnk in ["rank", "rank_gp", "rank_form", "rank_snap"]:
            record[rnk] = 0
        records.append(record)
    return records


def append_records(csv_dict, record_path, static_info):
    with open(record_path, "r") as f:
        record_dict = json.load(f, cls=NDArrayDecoder)
    save_props(record_dict, dirname(record_path))
    record_dict.update(static_info)
    records = get_records(record_dict)
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
            static_info = json.load(sf, cls=NDArrayDecoder)
            jplist = glob.glob(
                osjoin(dirname(static_path), "**/properties.json"), recursive=True
            )
            for record_path in jplist:
                append_records(csv_dict, record_path, static_info)

    df = pd.DataFrame(csv_dict)
    df["rank"] = df["accuracy"].rank(method="min", ascending=False)
    df["rank_gp"] = df.groupby("groupname")["accuracy"].rank("dense", ascending=False)
    df["rank_form"] = df.groupby("formname")["accuracy"].rank("dense", ascending=False)
    df["rank_snap"] = df.groupby(["run", "epoch"])["accuracy"].rank(
        "dense", ascending=False
    )
    df.to_csv(csv_path, header=True, index=False)
    write_to_sqldb(df, result_dir)


def write_to_sqldb(df, result_dir):
    print("Saving summary.csv as a SQL db")
    engine = sa.create_engine("sqlite:///{}".format(osjoin(result_dir, "summary.db")))
    df.to_sql("summary", engine, index=False, if_exists="replace")


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

    write_dict = {
        "rank": "global",
        "rank_gp": "in_group",
        "rank_form": "in_form",
        "rank_snap": "in_run",
    }
    stat_df = pd.DataFrame(
        index=list(write_dict.values()),
        columns=asc_cols[:-1] + ["out of"],
        dtype=np.int32,
    )
    dyn_df = pd.DataFrame(
        index=list(write_dict.values()),
        columns=["time"] + desc_cols + ["out of"],
        dtype=np.int32,
    )

    def get_ranks(df1, rownum):
        for k, v in write_dict.items():
            df1.loc[v] = ranking_dfs[k].loc[rownum, df1.columns]
        return df1.astype(np.int32).T.to_dict("split")

    rdf = ranking_dfs["rank"]
    for i in range(0, len(rdf), 4):
        stat_dict = get_ranks(stat_df, i)
        record_dict = {
            int(rdf.loc[i + j, "epoch"]): get_ranks(dyn_df, i + j) for j in range(4)
        }
        with open(osjoin(result_dir, rdf["name"][i], "rankings.json"), "w") as sf:
            json.dump(stat_dict, sf, cls=NDArrayEncoder)
        with open(
            osjoin(result_dir, rdf["name"][i], "runs", rdf["run"][i], "rankings.json"),
            "w",
        ) as rf:
            json.dump(record_dict, rf, cls=NDArrayEncoder)

    engine = sa.create_engine("sqlite:///{}".format(osjoin(result_dir, "rankings.db")))
    for k, v in ranking_dfs.items():
        v.to_csv(
            osjoin(result_dir, "rankings_{}.csv".format(write_dict[k])),
            header=True,
            index=False,
        )
        v.to_sql(write_dict[k], engine, index=False, if_exists="replace")
