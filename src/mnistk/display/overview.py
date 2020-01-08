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


def lv(label, value, vlabel=None):
    vlab = label if vlabel is None else vlabel
    return {"label": label, "value": "{}|{}".format(vlab, value)}


def filter_options(df):
    filter_opts = []
    for (r, e), _ in df.groupby(["run", "epoch"]):
        filter_opts.append(lv("run {}, epoch {}".format(r, e), "{}|{}".format(r, e)))
    default = filter_opts[0]["value"]
    dropdown = dcc.Dropdown(
        id="filter-options",
        options=filter_opts,
        multi=True,
        value=default,
        placeholder="Showing all runs; select run(s) here",
    )
    return dropdown, default


def grouping_options():
    group_opts = [lv("Class", "groupname"), lv("Activation Used", "activation")]
    default = group_opts[0]["value"]
    selector = dcc.RadioItems(id="grouping-options", options=group_opts, value=default)
    return selector, default


def xvalue_options():
    xval_opts = [
        lv("Time taken to train (s)", "time"),
        lv("Number of Parameters used", "#params"),
        lv("Approx. Memory Usage (bytes)", "memory per pass"),
        lv("Number of Layers", "#layers"),
        lv("Number of Operations", "#ops"),
    ]
    default = xval_opts[0]["value"]
    dropdown = dcc.Dropdown(
        id="perfx-dropdown", options=xval_opts, value=default, clearable=False
    )
    return dropdown, default


def yvalue_options():
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

    default = yval_opts[0]["value"]
    dropdown = dcc.Dropdown(
        id="perfy-dropdown", options=yval_opts, value=default, clearable=False
    )
    return dropdown, default


def figure_handler_closure(df_full, **defaults):
    def figure_handler(filter_opt, group_opt, xval, yval):
        if isinstance(filter_opt, str):
            filter_opt = [filter_opt]
        if len(filter_opt) == 0:
            df = df_full
        else:
            filter_opt = [x.split("|")[1:] for x in filter_opt]
            runs = [x[0] for x in filter_opt]
            epochs = [int(x[1]) for x in filter_opt]

            run_ind = [x in runs for x in df_full["run"]]
            ep_ind = [x in epochs for x in df_full["epoch"]]
            filter_ind = [x and ep_ind[i] for i, x in enumerate(run_ind)]
            df = df_full[filter_ind]

        gp_title, gp_opt = group_opt.split("|")
        xtitle, xcol = xval.split("|")
        ytitle, ycol = yval.split("|")
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

    fig = dcc.Graph(id="perf-graph", figure=figure_handler(**defaults))
    return fig, figure_handler


def get_application(csv_path):
    df = pd.read_csv(csv_path, header=0)
    external_css = ["https://cdnjs.cloudflare.com/ajax/libs/tufte-css/1.7.2/tufte.css"]
    app = dash.Dash("overview", external_stylesheets=external_css)
    set_app_layout(df, app)
    return app


def set_app_layout(df, app):
    x_select, x_default = xvalue_options()
    y_select, y_default = yvalue_options()
    filt_select, filt_default = filter_options(df)
    gp_select, gp_default = grouping_options()

    fig, figure_handler = figure_handler_closure(
        df,
        filter_opt=filt_default,
        group_opt=gp_default,
        xval=x_default,
        yval=y_default,
    )

    app.layout = html.Div(
        children=[
            html.H1("Hello Dash"),
            html.H2("View the performance of a thousand neural networks"),
            filt_select,
            gp_select,
            x_select,
            y_select,
            fig,
        ]
    )

    app.callback(
        dd.Output(component_id="perf-graph", component_property="figure"),
        [
            dd.Input(component_id="filter-options", component_property="value"),
            dd.Input(component_id="grouping-options", component_property="value"),
            dd.Input(component_id="perfx-dropdown", component_property="value"),
            dd.Input(component_id="perfy-dropdown", component_property="value"),
        ],
    )(figure_handler)
