# -*- coding: utf-8 -*-
"""
    mnistk.display.overview
    ~~~~~~~~~~~~~~~~~~~~~~~

    Show overview of network performance via Dash

    :copyright: (c) 2020 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
import pandas as pd


def plotmod_closure(df_full):
    def plot_modder(snap_opt, group_opt, xval, yval, *inputs):
        if isinstance(snap_opt, str):
            snap_opt = [snap_opt]
        if len(snap_opt) == 0:
            snap_opt = [
                lv("run {}, epoch {}".format(r, e), "{}|{}".format(r, e))["value"]
            ]
        snap_opt = [x.split("|")[1:] for x in snap_opt]
        runs = [x[0] for x in snap_opt]
        epochs = [int(x[1]) for x in snap_opt]

        a = [x in runs for x in df_full["run"]]
        b = [x in epochs for x in df_full["epoch"]]
        a = [x and b[i] for i, x in enumerate(a)]

        gp_title, gp_opt = group_opt.split("|")
        xtitle, xcol = xval.split("|")
        ytitle, ycol = yval.split("|")
        df = df_full[a]
        ans = {
            "data": [
                dict(
                    x=gp[xcol],
                    y=gp[ycol],
                    text=gp["name"],
                    mode="markers",
                    marker=dict(size=15),
                    opacity=0.6,
                    name=gname,
                )
                for gname, gp in df.groupby(gp_opt)
            ],
            "layout": dict(
                hovermode="closest",
                transition=dict(duration=700),
                xaxis={"title": xtitle},
                yaxis={"title": ytitle},
            ),
        }
        return ans

    return plot_modder


def lv(label, value, vlabel=None):
    vlab = label if vlabel is None else vlabel
    return {"label": label, "value": "{}|{}".format(vlab, value)}


def gdx_modder(value):
    return "The modding adds: {}".format(value)


def get_application(csv_path):
    df = pd.read_csv(csv_path, header=0)
    external_css = ["https://cdnjs.cloudflare.com/ajax/libs/tufte-css/1.7.2/tufte.css"]
    app = dash.Dash("overview", external_stylesheets=external_css)
    modder = plotmod_closure(df)

    snap_opts = []
    for (r, e), _ in df.groupby(["run", "epoch"]):
        snap_opts.append(lv("run {}, epoch {}".format(r, e), "{}|{}".format(r, e)))

    xval_opts = [
        lv("Time taken to train (s)", "time"),
        lv("Number of Parameters used", "#params"),
        lv("Approx. Memory Usage (bytes)", "memory per pass"),
        lv("Number of Layers", "#layers"),
        lv("Number of Operations", "#ops"),
    ]
    yval_opts = (
        [lv("Overall AUC", "AUC"), lv("Overall Accuracy", "accuracy")]
        + [
            lv(
                "{} pred. Accuracy".format(x),
                "accuracy_{}".format(x),
                "% correct {} predictions".format(x),
            )
            for x in range(10)
        ]
        + [
            lv(
                "{} pred. AUC".format(x),
                "AUC_{}".format(x),
                "AUC when predicting {}".format(x),
            )
            for x in range(10)
        ]
    )
    group_opts = [lv("Form Name", "formname"), lv("Group Name", "groupname")]

    app.layout = html.Div(
        children=[
            html.H1("Hello Dash"),
            html.H2("View the performance of a thousand neural networks"),
            dcc.Dropdown(
                id="snapshot-options",
                options=snap_opts,
                multi=True,
                value=snap_opts[0]["value"],
            ),
            dcc.RadioItems(
                id="grouping-options", options=group_opts, value=group_opts[0]["value"]
            ),
            dcc.Dropdown(
                id="xval-dropdown", options=xval_opts, value=xval_opts[0]["value"]
            ),
            dcc.Dropdown(
                id="yval-dropdown", options=yval_opts, value=yval_opts[0]["value"]
            ),
            dcc.Graph(
                id="perf-graph",
                figure=modder(
                    snap_opts[0]["value"],
                    lv("Form name", "formname")["value"],
                    lv("Time taken to train", "time")["value"],
                    lv("Overall AUC", "AUC")["value"],
                ),
            ),
        ]
    )

    app.callback(
        dd.Output(component_id="perf-graph", component_property="figure"),
        [
            dd.Input(component_id="snapshot-options", component_property="value"),
            dd.Input(component_id="grouping-options", component_property="value"),
            dd.Input(component_id="xval-dropdown", component_property="value"),
            dd.Input(component_id="yval-dropdown", component_property="value"),
        ],
    )(modder)

    return app
