# -*- coding: utf-8 -*-
"""
    mnistk.collect
    ~~~~~~~~~~~~~~

    Collect data from JSONs in subfolders into a CSV

    :copyright: (c) 2020 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from os.path import join as osjoin
import json
import glob
import pandas as pd


def column_names():
    cols = (
        [
            "name",
            "formname",
            "groupname",
            "run",
            "epoch",
            "#layers",
            "#params",
            "#ops",
            "time",
            "memory per pass",
            "loss",
        ]
        + ["AUC"]
        + ["AUC_{}".format(x) for x in range(10)]
        + ["accuracy"]
        + ["accuracy_{}".format(x) for x in range(10)]
    )
    return cols


def get_groupname(name):
    for x in ["Basic", "ResNet", "Conv1d", "Conv2d", "Conv3d", "Linear"]:
        if x in name:
            return x
    return name


def append_json(csv_dict, json_path):
    with open(json_path) as f:
        jdict = json.load(f)

        for epoch in jdict["test loss"].keys():
            csv_dict["epoch"].append(int(epoch))
            csv_dict["formname"].append(jdict["name"].split("_")[0])
            csv_dict["groupname"].append(get_groupname(jdict["name"]))
            for k, v in jdict.items():
                if k in ["test AUC", "test accuracy"]:
                    modkey = k.replace("test ", "")
                    csv_dict[modkey].append(sum(jdict[k][epoch]) / 10)
                    for x in range(10):
                        csv_dict["{}_{}".format(modkey, x)].append(jdict[k][epoch][x])
                elif k == "test loss":
                    modkey = k.replace("test ", "")
                    csv_dict[modkey].append(jdict[k][epoch])
                elif k == "time per epoch":
                    csv_dict["time"].append(jdict[k] * int(epoch))
                elif k == "train loss":
                    continue
                else:
                    csv_dict[k].append(jdict[k])


def write_to_csv(result_dir, csv_path):
    colnames = column_names()

    csv_dict = {col: [] for col in colnames}
    jplist = glob.glob(osjoin(result_dir, "**/properties.json"), recursive=True)
    for json_path in jplist:
        append_json(csv_dict, json_path)

    df = pd.DataFrame(csv_dict)
    df.to_csv(csv_path, header=True, index=False)
    max_rows = len(csv_dict["run"])
