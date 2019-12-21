# -*- coding: utf-8 -*-
"""
    mnistk.run.settings
    ~~~~~~~~~~~~~~~~~

    Extracts settings from JSON

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import json
from collections import namedtuple

Settings = namedtuple(
    "Settings",
    (
        "epochs",
        "test_every",  # k epochs
        "train_batch_size",
        "test_batch_size",
        "hypers",
        "use_gpu",
        "plot_structure",
        "save_predictions",
        "save_snapshot",
        "plot_losses",
    ),
)

Settings.__new__.__defaults__ = (
    10,
    10,
    10,
    10,
    dict(lr=0.001),
    False,
    False,
    False,
    False,
    False,
)


def print_settings(self):
    return "\n".join(
        ["Settings:"] + ["\t{} = {}".format(k, v) for k, v in self._asdict().items()]
    )


Settings.__repr__ = print_settings
