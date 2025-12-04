

"""
This small helper function handles exporting data that the user is
currently looking at in the Dashboard.

When the user clicks the Export button, we just filter the DataFrame using the
selected agents/models and send it back as a downloadable CSV.
"""

from dash import dcc, no_update

def export_filtered_csv(df, agents, models, ctx, n_clicks):
    # Only export when the Export button is the thing that triggered the callback
    if not ctx.triggered_id == "export-btn":
        return no_update
    # Apply the current Dashboard filters
    filtered = df[df["agent"].isin(agents) & df["model"].isin(models)]

    # Return a downloadable CSV
    return dcc.send_data_frame(filtered.to_csv, "filtered_results.csv", index=False)
