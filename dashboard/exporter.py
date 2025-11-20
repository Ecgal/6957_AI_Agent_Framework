from dash import dcc, no_update

def export_filtered_csv(df, agents, models, ctx, n_clicks):
    from dash import no_update
    if not ctx.triggered_id == "export-btn":
        return no_update
    filtered = df[df["agent"].isin(agents) & df["model"].isin(models)]
    return dcc.send_data_frame(filtered.to_csv, "filtered_results.csv", index=False)
