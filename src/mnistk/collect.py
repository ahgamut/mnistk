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
import csv


def column_names():
    cols = (
        [
            "name",
            "#layers",
            "#params",
            "run",
            "time",
            "memory per pass",
            "epoch",
            "loss",
        ]
        + ["AUC"]
        + ["AUC_{}".format(x) for x in range(10)]
        + ["accuracy"]
        + ["accuracy_{}".format(x) for x in range(10)]
    )
    return cols


def append_json(csv_dict, json_path):
    with open(json_path) as f:
        jdict = json.load(f)

        for epoch in jdict["test loss"].keys():
            csv_dict["epoch"].append(int(epoch))
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
    with open(csv_path, "w", newline="") as csv_file:
        colnames = column_names()
        writer = csv.DictWriter(csv_file, fieldnames=colnames)

        csv_dict = {col: [] for col in colnames}
        jplist = glob.glob(osjoin(result_dir, "**/properties.json"), recursive=True)
        for json_path in jplist:
            append_json(csv_dict, json_path)
        max_rows = len(csv_dict["run"])

        writer.writeheader()
        for i in range(max_rows):
            writer.writerow({col: csv_dict[col][i] for col in colnames})
