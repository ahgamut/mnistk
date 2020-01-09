# -*- coding: utf-8 -*-
"""
    mnistk.run.trainer
    ~~~~~~~~~~~~~~~~

    Controller class for generating data,
    training the networks, saving required info, etc.

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import sys, os
import shutil
import numpy as np
import time
import torch
from mnistk.run import DataGen, construct
from mnistk.run.utils import (
    get_recorder,
    approx_mem_usage,
    layerop_count,
    plot_structure,
    get_classwise_preds,
    save_props,
    save_predictions,
)
from mnistk.run.loss import LossFunc
from mnistk.run.optimizer import get_optimizer


class Trainer(object):

    """
    Object class to handle the initialization,
    training, testing, and saving of the details
    for every run
    """

    def __init__(self, settings, dest, net_id=0, run_name="t1"):
        self._settings = settings
        self.datagen = DataGen(settings.train_batch_size, settings.test_batch_size)
        self.net = construct(net_id)

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and self._settings.use_gpu:
            self.device = torch.device("cuda")
            print("CUDA available, using GPU for run")
        self.net = self.net.to(self.device)
        self.loss_fn = LossFunc()
        self.optimizer = get_optimizer(**settings.hypers)(self.net.parameters())

        self.run_name = run_name
        self._dest = os.path.join(dest, str(self.net.__class__.__name__).split(".")[-1])
        runs = os.path.join(self._dest, "runs/")
        this_run = os.path.join(runs, run_name)
        if not os.path.exists(self._dest):
            os.makedirs(self._dest)
        if os.path.exists(this_run) and os.path.isdir(this_run):
            shutil.rmtree(this_run)
        os.makedirs(this_run)
        self.run_dest = this_run
        self.state_props()

    def summary(self):
        print("Destination Folder:", self._dest)
        print(self._settings)
        print("Network:\n\t", self.net)

    def state_props(self):
        self.net_state = dict()
        self.net_state["train loss"] = []
        self.net_state["run"] = self.run_name
        self.net_state["time per epoch"] = 0
        self.net_state["test loss"] = {}
        self.net_state["test AUC"] = {}
        self.net_state["test accuracy"] = {}

    def train(self):
        self.net_state["train loss"] = []
        time_spent = []
        ep_start = None

        self.net.train()
        for i in range(1, self._settings.epochs + 1):
            ep_losses = []
            ep_start = time.time()
            for x_d, y_d in self.datagen.train():
                x, y_true = x_d.to(self.device), y_d.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.net(x)
                loss = self.loss_fn(y_true, y_pred)
                loss.backward()
                self.optimizer.step()
                ep_losses.append(loss.item())
            time_spent.append(time.time() - ep_start)
            self.net_state["train loss"].append(np.mean(ep_losses))
            print(
                "Epoch {:4}: Training Loss = {:1.3f}".format(
                    i, self.net_state["train loss"][-1]
                ),
                file=sys.stderr,
            )
            if i % self._settings.test_every == 0:
                self.test(epoch=i)
        self.net_state["time per epoch"] = np.mean(time_spent)

    def test(self, epoch):
        self.net.eval()
        ep_losses = []

        preds = []
        trues = []
        for x_d, y_d in self.datagen.test():
            x, y_true = x_d.to(self.device), y_d.to(self.device)
            y_pred = self.net(x)
            loss = self.loss_fn(y_true, y_pred)
            preds.append(y_pred.detach().cpu())
            trues.append(y_true.detach().cpu())
            ep_losses.append(loss.item())

        loss = np.mean(ep_losses)
        trues = torch.cat(trues).numpy()
        preds = torch.cat(preds).numpy()

        print("Epoch {:4}: Test Loss = {:1.3f}".format(epoch, loss))
        if self._settings.save_predictions:
            aucs, accus = get_classwise_preds(trues, preds)
            self.net_state["test loss"][epoch] = loss
            self.net_state["test AUC"][epoch] = aucs
            self.net_state["test accuracy"][epoch] = accus
            print(
                "Epoch {:4}: Test Accuracy = {:2.1f}%".format(
                    epoch, 100 * np.mean(accus)
                ),
                file=sys.stderr,
            )
            save_predictions(preds, self._dest, self.run_name, epoch)

        if self._settings.save_snapshot:
            torch.save(
                self.net.state_dict(),
                os.path.join(self.run_dest, "network-{}.pth".format(epoch)),
            )

        self.net.train()

    def save_static_props(self):
        # save properties that are same across all runs
        static_info = dict()
        static_info["name"] = self.net.__class__.__name__
        static_info["#layers"] = len(list(self.net.children()))
        static_info["#ops"] = 0
        static_info["memory per pass"] = 0
        static_info["activation"] = "None"
        if "ResNet" in self.net.__class__.__name__:
            static_info["activation"] = "ReLU"
        else:
            for x in self.net.children():
                if (
                    "activation" in x.__module__
                    and x.__class__.__name__ != "LogSoftmax"
                ):
                    static_info["activation"] = x.__class__.__name__
        static_info["#params"] = sum(
            q.numel() for q in self.net.parameters(recurse=True) if q.requires_grad
        )
        rec = get_recorder(self.net.cpu(), tuple([1] + list(self.net.in_shape)))
        if rec is not None:
            print("Plotting Network structure as SVG...", file=sys.stderr)
            plot_structure(rec, directory=self._dest, fmt="svg")
            static_info["#ops"], static_info["#layers"] = layerop_count(rec)
            static_info["memory per pass"] = approx_mem_usage(rec)
            del rec
        save_props(static_info, self._dest, "network.json")

    def save(self):
        print("Saving properties...", file=sys.stderr)
        if self._settings.save_static_info:
            self.save_static_props()
        save_props(self.net_state, self.run_dest)
