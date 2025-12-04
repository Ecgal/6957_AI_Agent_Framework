

"""
This file builds the entire web Dashboard used to visualize the results of the
AI agent attack tests. All of the metrics generated during the runs get loaded
here, filtered, and then turned into the different summary charts.

The Dashboard is meant to be a quick way to compare:
- performance across agents and models
- how each attack type behaves on different pages/environments
- trends over time as more benchmarking runs are added

A few important notes about how this file is structured:

1. We load all results once at startup, since the Dashboard is only meant for
   post run. *** Live result will be added soon **

2. The layout is divided into clear sections (Agent/Model, Page/Env, Trends,
   and the DataTable).

3. All of the actual chart building logic lives in charts.py. This file only
   handles wiring things together and telling Dash which figures go where.

4. Two callbacks power the whole app:
   - update_dashboard(): regenerates the charts whenever the user changes the
     selected agents/models.
   - handle_export(): lets the user export whatever slice of the data they're
     currently viewing.

"""




from dash import Dash, html, dcc, Input, Output, dash_table, ctx
from data_loader import load_results
from exporter import export_filtered_csv
import plotly.io as pio
from charts import generate_charts, make_page_env_heatmap, make_recent_vs_all_by_attack, make_model_time_trend, make_agent_model_time_trend

df = load_results()
if df.empty:
    raise SystemExit("No result files found.")

pio.templates["invrsn_dark"] = pio.templates["plotly_dark"]

pio.templates["invrsn_dark"].layout.update(
    paper_bgcolor="#1e1f23",     # card background color
    plot_bgcolor="#1e1f23",
    font=dict(color="#eaeaea"),
    xaxis=dict(gridcolor="#2a2c32"),
    yaxis=dict(gridcolor="#2a2c32"),
)

update_charts = generate_charts(df)

app = Dash(__name__, title="Attack Summary")

app.layout = html.Div([
    html.H1("AI Agent Attack Performance Summary",
            style={"textAlign": "center", "marginBottom": "30px"}),


    # FILTERS & EXPORT
    html.Div([
        html.Div([
            html.Label("Select Agents:", className="dropdown-col"),
            dcc.Dropdown(
                df["agent"].unique(),
                df["agent"].unique(),
                multi=True,
                id="agent-select",
                className="dash-dropdown"
            ),
        ], style={"width": "45%", "display": "inline-block"}),

        html.Div([
            html.Label("Select Models:", className="dropdown-col"),
            dcc.Dropdown(
                df["model"].unique(),
                df["model"].unique(),
                multi=True,
                id="model-select",
                className="dash-dropdown"
            ),
        ], style={"width": "45%", "marginLeft": "5%", "display": "inline-block"}),
    ], className="dropdown-row"),

    html.Div([
        html.Button("️ Export Summary to CSV", id="export-btn", n_clicks=0, className="dash-button"),
        dcc.Download(id="download-dataframe-csv"),
    ], style={"marginBottom": "40px"}),

    # SECTION 1: AGENT + MODEL PERFORMANCE
    html.Div([
        html.H2("Agent & Model Performance", className="section-title"),
        html.Div([
            html.Div(dcc.Graph(id="agent-bar"), className="half-card"),
            html.Div(dcc.Graph(id="model-bar"), className="half-card"),
        ], className="row"),

        html.Div([
            html.Div(dcc.Graph(id="agent-model-bar"), className="full-card"),
        ])
    ]),

    # SECTION 2: PAGE & ENVIRONMENT
    html.Div([
        html.H2("Attack Performance by Page & Environment", className="section-title"),
        html.Div([
            html.Div(dcc.Graph(id="page-bar"), className="half-card"),
            html.Div(dcc.Graph(id="env-bar"), className="half-card"),
        ], className="row"),

        html.Div([
            html.Div(dcc.Graph(id="page-env-heatmap"), className="full-card"),
        ]),
    ]),

    # SECTION 3: COMPARISON AND TIME
    html.Div([
        html.H2("Performance Trends & Comparison", className="section-title"),
        html.Div([
            html.Div(dcc.Graph(id="compare-bar"), className="full-card"),
            html.Div(dcc.Graph(id="recent-vs-all-by-attack"), className="full-card"),
        ], className="row"),

        html.Div([
            html.Div(dcc.Graph(id="time-trend"), className="full-card"),
            html.Div(dcc.Graph(id="model-time-trend"), className="half-card"),
            html.Div(dcc.Graph(id="agent-model-time-trend"), className="half-card"),
        ], className="row"),
    ]),

    # === SECTION 4: DATA TABLE ===
    html.Div([
        html.H2("Run Summary Table", className="section-title"),
        dash_table.DataTable(
            id="summary-table",
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict("records"),
            page_size=10,
            style_table={"overflowX": "auto"},
            style_header={
                "fontWeight": "bold",
                "backgroundColor": "#2a2b30",
                "color": "#ffffff"
            },
            style_cell={
                "textAlign": "left",
                "padding": "8px",
                "fontSize": "13px",
                "backgroundColor": "#1e1f23",
                "color": "#eaeaea"
            },
        ),
    ], className="data-table", style={"marginTop": "40px"}),

], style={"maxWidth": "1200px", "margin": "auto", "padding": "20px 20px 80px 20px"})




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
