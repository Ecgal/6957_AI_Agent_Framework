import pandas as pd
import plotly.express as px


def generate_charts(df):
    def update_charts(agents, models):
        filtered = df[df["agent"].isin(agents) & df["model"].isin(models)]

        latest_per_agent = filtered.groupby("agent")["timestamp"].max().reset_index()
        latest_df = pd.merge(filtered, latest_per_agent, on=["agent", "timestamp"], how="inner")

        # === Comparison Chart (All vs Most Recent) ===
        comparison_df = pd.concat([
            filtered.groupby("agent")["success_rate"].mean().reset_index().assign(run="All Runs"),
            latest_df.groupby("agent")["success_rate"].mean().reset_index().assign(run="Most Recent Run")
        ])
        fig_compare = px.bar(
            comparison_df, x="agent", y="success_rate", color="run", barmode="group",
            title="Most Recent Run vs All Runs (Avg Success Rate)"
        )
        fig_compare.update_layout(
            xaxis_title="Agent",
            yaxis_title="Average Success Rate (%)",
            template="invrsn_dark"
        )

        # === Agent Average ===
        fig_agent = px.bar(
            filtered.groupby("agent")["success_rate"].mean().reset_index(),
            x="agent", y="success_rate", color="agent",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Average Attack Success Rate per Agent"
        )
        fig_agent.update_layout(
            xaxis_title="Agent",
            yaxis_title="Average Success Rate (%)",
            template="invrsn_dark"
        )

        # === Model Average ===
        fig_model = px.bar(
            filtered.groupby("model")["success_rate"].mean().reset_index(),
            x="model", y="success_rate", color="model",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Average Attack Success Rate per Model"
        )
        fig_model.update_layout(
            xaxis_title="Model",
            yaxis_title="Average Success Rate (%)",
            template="invrsn_dark"
        )

        # === Agent + Model Combined ===
        fig_agent_model = px.bar(
            filtered.groupby(["agent", "model"])["success_rate"].mean().reset_index(),
            x="agent", y="success_rate", color="model", barmode="group",
            title="Average Attack Success Rate by Agent and Model"
        )
        fig_agent_model.update_layout(
            xaxis_title="Agent",
            yaxis_title="Average Success Rate (%)",
            template="invrsn_dark"
        )

        # === Attack Type ===
        fig_attack = px.bar(
            filtered.groupby("page")["success_rate"].mean().reset_index(),
            x="page", y="success_rate", color_discrete_sequence=["#636EFA"],
            title="Average Success Rate per Attack Type",
            labels={"page": "Attack Type", "success_rate": "Avg Success Rate"}
        )
        fig_attack.update_layout(
            xaxis_title="Attack Type",
            yaxis_title="Average Success Rate (%)",
            template="invrsn_dark"
        )

        # === Environment ===
        if "env" in filtered.columns:
            fig_env = px.bar(
                filtered.groupby("env")["success_rate"].mean().reset_index(),
                x="env", y="success_rate", color_discrete_sequence=["#00CC96"],
                title="Average Attack Success Rate per Environment"
            )
            fig_env.update_layout(
                xaxis_title="Environment",
                yaxis_title="Average Success Rate (%)",
                template="invrsn_dark"
            )
        else:
            fig_env = px.bar(title="No environment data")

        # === Time Trend ===
        filtered = filtered.assign(
            date=pd.to_datetime(filtered["timestamp"], format='ISO8601', errors="coerce", utc=True)
            .dt.tz_convert(None)
            .dt.date
        ).dropna(subset=["date"])

        fig_time = px.line(
            filtered.groupby(["date", "agent"])["success_rate"].mean().reset_index(),
            x="date", y="success_rate", color="agent", markers=True,
            title="Agent Performance Over Time (Daily Average)",
            labels={"date": "Date", "success_rate": "Avg Success Rate"}
        )
        fig_time.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Success Rate (%)",
            template="invrsn_dark"
        )

        return fig_agent, fig_model, fig_agent_model, fig_attack, fig_env, fig_time, fig_compare

    return update_charts

def make_page_env_heatmap(df):
    pivot = (
        df.groupby(["page", "env"])["success_rate"]
          .mean()
          .reset_index()
          .pivot(index="page", columns="env", values="success_rate")
          .fillna(0)
    )

    fig_heatmap = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="Viridis",
        title="Average Attack Success Rate per Page per Environment",
        labels=dict(x="Environment", y="Page", color="Success Rate (%)")
    )
    fig_heatmap.update_xaxes(
        side="bottom",
        tickangle=30,
        tickfont=dict(size=10),
    )

    fig_heatmap.update_layout(
        margin=dict(l=80, r=80, t=80, b=140),
        template="invrsn_dark"
    )
    return fig_heatmap


def make_recent_vs_all_by_attack(df):
    if df.empty or "timestamp" not in df.columns:
        return px.bar(title="No data available")

    # Get most recent timestamp per (agent, model)
    latest_per_combo = df.groupby(["agent", "model"])["timestamp"].max().reset_index()
    latest_df = pd.merge(df, latest_per_combo, on=["agent", "model", "timestamp"], how="inner")

    # Aggregate success rates
    historical = (
        df.groupby(["agent", "model", "page"])["success_rate"]
          .mean()
          .reset_index()
          .assign(run="All Runs")
    )
    latest = (
        latest_df.groupby(["agent", "model", "page"])["success_rate"]
          .mean()
          .reset_index()
          .assign(run="Most Recent Run")
    )

    comparison = pd.concat([historical, latest])

    fig = px.bar(
        comparison,
        x="page",
        y="success_rate",
        color="model",               # MODEL = color
        pattern_shape="run",         # RUN = pattern
        barmode="group",
        facet_col="agent",           # single row of agents
        title="Most Recent Run vs All Runs by Attack Type (Per Agent)",
        labels={"page": "Attack Type", "success_rate": "Avg Success Rate"}
    )

    fig.update_xaxes(
        tickangle=30,
        automargin=True,
        title=None
    )
    fig.update_yaxes(automargin=True)

    fig.for_each_annotation(lambda a: a.update(y=a.y + 0.05))

    fig.update_layout(
        height=450,
        margin=dict(l=40, r=40, t=80, b=120),
        legend_title="Legend",
        template="invrsn_dark"
    )
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[1],
        font=dict(size=14, color="#eaeaea"),

    ))

    return fig



def make_model_time_trend(df):
    """
    Show average model performance over time (daily average success rate).
    """
    df = df.assign(
        date=pd.to_datetime(df["timestamp"], format='ISO8601', errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.date
    ).dropna(subset=["date"])

    trend = (
        df.groupby(["date", "model"])["success_rate"]
          .mean()
          .reset_index()
    )

    fig = px.line(
        trend,
        x="date", y="success_rate", color="model", markers=True,
        title="Model Performance Over Time (Daily Average)",
        labels={"date": "Date", "success_rate": "Avg Success Rate"}
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average Success Rate",
        hovermode="x unified",
        legend_title="Model",
        template="invrsn_dark"
    )
    return fig

def make_agent_model_time_trend(df):
    df = df.assign(
        date=pd.to_datetime(df["timestamp"], format='ISO8601', errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.date
    ).dropna(subset=["date"])

    trend = (
        df.groupby(["date", "agent", "model"])["success_rate"]
          .mean()
          .reset_index()
    )

    trend["agent_model"] = trend["agent"] + " | " + trend["model"]

    fig = px.line(
        trend,
        x="date",
        y="success_rate",
        color="agent_model",
        markers=True,
        title="Agent + Model Performance Over Time (Daily Average)",
        labels={
            "date": "Date",
            "success_rate": "Avg Success Rate",
            "agent_model": "Agent | Model"
        }
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average Success Rate",
        hovermode="x unified",
        legend_title="Agent | Model",
        height=500,
        margin=dict(l=50, r=50, t=60, b=50),
        template="invrsn_dark"
    )

    return fig
