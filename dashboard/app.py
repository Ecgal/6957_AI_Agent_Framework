from dash import Dash, html, dcc, Input, Output, dash_table, ctx
from data_loader import load_results
from exporter import export_filtered_csv
from charts import generate_charts, make_page_env_heatmap, make_recent_vs_all_by_attack, make_model_time_trend, \
    make_agent_model_time_trend

df = load_results()
if df.empty:
    raise SystemExit("No result files found.")

update_charts = generate_charts(df)

app = Dash(__name__, title="Attack Summary")

app.layout = html.Div([
    html.H2("Agent Attack Performance Summary"),

    html.Div([
        html.Label("Select Agents:"),
        dcc.Dropdown(df["agent"].unique(), df["agent"].unique(), multi=True, id="agent-select")
    ], style={"width": "40%", "display": "inline-block"}),

    html.Div([
        html.Label("Select Models:"),
        dcc.Dropdown(df["model"].unique(), df["model"].unique(), multi=True, id="model-select")
    ], style={"width": "40%", "marginLeft": "5%", "display": "inline-block"}),

    html.Br(),
    html.Button("Export Run Summary to CSV", id="export-btn", n_clicks=0),
    dcc.Download(id="download-dataframe-csv"),
    html.Br(), html.Br(),

    dcc.Graph(id="agent-bar"),
    dcc.Graph(id="model-bar"),
    dcc.Graph(id="agent-model-bar"),
    dcc.Graph(id="page-bar"),
    dcc.Graph(id="env-bar"),
    dcc.Graph(id="page-env-heatmap"),
    dcc.Graph(id="time-trend"),
    dcc.Graph(id="compare-bar"),
    dcc.Graph(id="recent-vs-all-by-attack"),
    dcc.Graph(id="model-time-trend"),
    dcc.Graph(id="agent-model-time-trend"),


    html.H3("Run Summary Table"),
    dash_table.DataTable(
        id="summary-table",
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("records"),
        page_size=10,
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "bold"},
    ),
])

@app.callback(
    Output("agent-bar", "figure"),
    Output("model-bar", "figure"),
    Output("agent-model-bar", "figure"),
    Output("page-bar", "figure"),
    Output("env-bar", "figure"),
    Output("summary-table", "data"),
    Output("time-trend", "figure"),
    Output("compare-bar", "figure"),
    Output("page-env-heatmap", "figure"),
    Output("recent-vs-all-by-attack", "figure"),
    Output("model-time-trend", "figure"),
    Output("agent-model-time-trend", "figure"),
    Input("agent-select", "value"),
    Input("model-select", "value")
)
def update_dashboard(agents, models):
    figs = update_charts(agents, models)
    filtered = df[df["agent"].isin(agents) & df["model"].isin(models)]
    heatmap = make_page_env_heatmap(filtered)
    comparison_fig = make_recent_vs_all_by_attack(filtered)
    fig_model_time = make_model_time_trend(filtered)
    fig_agent_model_time = make_agent_model_time_trend(filtered)
    return (*figs[:5], filtered.to_dict("records"), *figs[5:],heatmap,comparison_fig,fig_model_time, fig_agent_model_time)

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-btn", "n_clicks"),
    Input("agent-select", "value"),
    Input("model-select", "value")
)
def handle_export(n_clicks, agents, models):
    return export_filtered_csv(df, agents, models, ctx, n_clicks)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
