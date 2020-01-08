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
import json
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from mnistk.run import DataGen, construct
from mnistk.run.utils import (
    plot_structure,
    plot_predictions,
    save_props,
    plot_losses,
    save_predictions,
    plot_images,
    approx_mem_usage,
    op_count,
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

        self.net_state = dict()
        self.run_name = run_name
        self.train_losses = []
        self.samples = []
        self.start_props()
        self.class_samples()

        self._dest = os.path.join(dest, str(self.net.__class__.__name__).split(".")[-1])
        self.run_dest = None
        runs = os.path.join(self._dest, "runs/")
        this_run = os.path.join(runs, run_name)
        if not os.path.exists(self._dest):
            os.makedirs(self._dest)
        if os.path.exists(this_run) and os.path.isdir(this_run):
            shutil.rmtree(this_run)
        os.makedirs(this_run)
        self.run_dest = this_run
        self.writer = SummaryWriter(log_dir=this_run)

    def summary(self):
        print("Destination Folder:", self._dest)
        print(self._settings)
        print("Network:\n\t", self.net)

    def start_props(self):
        self.net_state["name"] = self.net.__class__.__name__
        self.net_state["activation"] = "None"
        for x in self.net.children():
            if "activation" in x.__module__ and x.__class__.__name__ != "LogSoftmax":
                self.net_state["activation"] = x.__class__.__name__
        self.net_state["#layers"] = len(list(self.net.children()))
        self.net_state["#params"] = sum(
            q.numel() for q in self.net.parameters(recurse=True) if q.requires_grad
        )
        self.net_state["#ops"] = 0
        self.net_state["run"] = self.run_name
        self.net_state["time per epoch"] = 0
        self.net_state["memory per pass"] = 0
        self.net_state["test loss"] = {}
        self.net_state["test AUC"] = {}
        self.net_state["test accuracy"] = {}

    def train(self):
        self.train_losses = []
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
            loss = np.mean(ep_losses)
            self.writer.add_scalar(tag="train/loss", scalar_value=loss, global_step=i)
            self.train_losses.append(loss)
            print(
                "Epoch {:4}: Training Loss = {:1.3f}".format(i, loss), file=sys.stderr
            )
            if i % self._settings.test_every == 0:
                self.test(epoch=i)
        self.net_state["time per epoch"] = np.mean(time_spent)

        print("\n")

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
            aucs, accu = plot_predictions(
                epoch, loss, trues, preds, self.run_dest, writer=self.writer
            )
            self.net_state["test loss"][epoch] = loss
            self.net_state["test AUC"][epoch] = aucs
            self.net_state["test accuracy"][epoch] = accu
            print(
                "Epoch {:4}: Test Accuracy = {:2.1f}%".format(
                    epoch, 100 * np.mean(accu)
                ),
                file=sys.stderr,
            )
            save_predictions(preds, self._dest, self.run_name, epoch)
            plot_images(epoch, *self.get_backprops(), self.run_dest, writer=self.writer)

        if self._settings.save_snapshot:
            torch.save(
                self.net.state_dict(),
                os.path.join(self.run_dest, "network-{}.pth".format(epoch)),
            )

        self.net.train()

    def class_samples(self):
        count = 0
        samples = {}
        test = self.datagen.test()
        for xt, yt in test:
            for i, y in enumerate(yt):
                if samples.get(y.item(), None) is None:
                    samples[y.item()] = (y, xt[i])
                    count += 1
            if count == 10:
                break
        self.samples = sorted(samples.items(), key=lambda x: x[0])
        self.samples = [x[1] for x in self.samples]

    def get_backprops(self):
        inps = []
        outs = []
        for yp, xp in self.samples:
            x, y_true = (
                xp.unsqueeze(0).to(self.device),
                yp.unsqueeze(0).to(self.device),
            )
            x.requires_grad_(True)
            y_pred = self.net(x)
            inps.append(x)
            outs.append(y_pred)
        return inps, outs

    def save(self):
        self.net_state["train loss"] = self.train_losses
        self.writer.add_graph(
            model=self.net, input_to_model=(self.samples[0][1].unsqueeze(0),)
        )

        if self._settings.plot_structure:
            print("Plotting Network structure as SVG...", file=sys.stderr)
            rec = plot_structure(
                self.net,
                tuple([1] + list(self.net.in_shape)),
                directory=self._dest,
                fmt="svg",
            )
            if rec is not None:
                self.net_state["#ops"] = op_count(rec)
                self.net_state["memory per pass"] = approx_mem_usage(rec)
                del rec
        if self._settings.plot_losses:
            print("Plotting training losses...", file=sys.stderr)
            plot_losses(
                self.net_state["train loss"], self.net_state["test loss"], self.run_dest
            )

        print("Saving properties...", file=sys.stderr)
        save_props(self.net_state, self.run_dest)
