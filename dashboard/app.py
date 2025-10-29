from collections import defaultdict
import dash
from dash import html, dcc, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import json, glob
from dash import ctx


# want stylize this better
# possibly use sqllite, see what group found


#using dash currently, I like the ease of use and customizable nature of it
#loads data in python
#sets a react virtual DOM, Plotly for charts, starts a server with flask
#callbacks sections if for updating the chart elements when a user interacts with it

#For example if we only want to see BrowserUse results, we can update the charts



#things to add,
# - better reporting for environments ( specific attacks
# - trends over time, how agents/model perform historically





#load json from the results
def load_results(results_dir="results"):
    files = glob.glob(f"../{results_dir}/results_*.json")
    records = []

    for f in files:
        try:
            with open(f) as infile:
                data = json.load(infile)
                agent = data.get("agent", "unknown")
                model = data.get("model", "unknown")
                timestamp = data.get("timestamp", "")

                # Map pages to the environments
                page_to_envs = defaultdict(set)
                all_envs = set()
                for r in data.get("results", []):
                    page_to_envs[r.get("page")].add(r.get("env"))
                    all_envs.add(r.get("env"))

                # Attach environments and the summaries
                for s in data.get("summary", []):
                    envs = page_to_envs.get(s["page"], {"unknown"})
                    for env in envs:
                        record = s.copy()
                        record.update({
                            "agent": agent,
                            "model": model,
                            "timestamp": timestamp,
                            "env": env
                        })
                        records.append(record)

                # add envs with no summary success data
                summarized_envs = {r["env"] for r in records if "env" in r}
                for missing_env in (all_envs - summarized_envs):
                    records.append({
                        "page": "N/A",
                        "success_rate": 0.0,
                        "successes": 0,
                        "total": 0,
                        "agent": agent,
                        "model": model,
                        "timestamp": timestamp,
                        "env": missing_env
                    })

        except Exception as e:
            print(f"Skipped {f}: {e}")

    return pd.DataFrame(records)


# load our data
df = load_results()
if df.empty:
    raise SystemExit("No result files found in /results")




# dashboard items
app = dash.Dash(__name__, title="Attack Summary")
app.layout = html.Div([
    html.H2(" Agent Attack Performance Summary"),

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

    html.Br(),
    html.Br(),

    html.H3(" Attack Success Rate Breakdown"),
    dcc.Graph(id="agent-bar"),
    dcc.Graph(id="model-bar"),
    dcc.Graph(id="agent-model-bar"),
    dcc.Graph(id="page-bar"),
    dcc.Graph(id="env-bar"),

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
    Input("agent-select", "value"),
    Input("model-select", "value"),
)
def update_charts(agents, models):
    filtered = df[df["agent"].isin(agents) & df["model"].isin(models)]

    # Per agent attack success rate
    fig_agent = px.bar(
        filtered.groupby("agent")["success_rate"].mean().reset_index(),
        x="agent", y="success_rate", color="agent",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Average Attack Success Rate per Agent"
    )

    # Per model attack success rate
    fig_model = px.bar(
        filtered.groupby("model")["success_rate"].mean().reset_index(),
        x="model", y="success_rate", color="model",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Average Attack Success Rate per Model"
    )

    # agent and model attack rate
    fig_agent_model = px.bar(
        filtered.groupby(["agent","model"])["success_rate"].mean().reset_index(),
        x="agent", y="success_rate", color="model",
        barmode="group",
        title="Average Attack Success Rate by Agent and Model"
    )

    # Per attack type success rate
    fig_page = px.bar(
        filtered.groupby("page")["success_rate"].mean().reset_index(),
        x="page", y="success_rate", color_discrete_sequence=["#636EFA"],
        title="Average Attack Success Rate per Attack type"
    )

    # Per environment success rate
    if "env" in filtered.columns and filtered["env"].notna().any():
        fig_env = px.bar(
            filtered.groupby("env")["success_rate"].mean().reset_index(),
            x="env", y="success_rate", color_discrete_sequence=["#00CC96"],
            title="Average Attack Success Rate per Environment"
        )
    else:
        fig_env = px.bar(title="No environment data in current results")

    return fig_agent, fig_model, fig_agent_model, fig_page, fig_env, filtered.to_dict("records")


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-btn", "n_clicks"),
    Input("agent-select", "value"),
    Input("model-select", "value"),
    prevent_initial_call=True,
)
def export_filtered_csv(n_clicks, agents, models):
    if not ctx.triggered_id == "export-btn":
        return dash.no_update

    filtered = df[df["agent"].isin(agents) & df["model"].isin(models)]
    return dcc.send_data_frame(filtered.to_csv, "filtered_results.csv", index=False)


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
