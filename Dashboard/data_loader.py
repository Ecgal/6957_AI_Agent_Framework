
"""
Data Loader for Dashboard + Reporting

This file handles pulling together all of the saved result JSON files from
the /results directory and turning them into a single Pandas DataFrame that
the Dashboard and exporter use. Each results_*.json file represents one full
benchmarking run for a specific agent/model pair.

The structure of the JSON isn’t perfectly flat, so this loader does a few
extra things:
- maps pages to environments so we can join summary metrics back to the env
- expands summary rows so each environment gets represented correctly
- fills in missing environments with a default "no results" row so charts
  don't break or drop categories
"""

from collections import defaultdict
import json, glob
import pandas as pd

def load_results(results_dir="results"):
    # Load all results_*.json files from the results directory
    files = glob.glob(f"../{results_dir}/results_*.json")
    records = []

    for f in files:
        try:
            with open(f) as infile:
                data = json.load(infile)

                # Basic metadata for the whole run
                agent = data.get("agent", "unknown")
                model = data.get("model", "unknown")
                timestamp = data.get("timestamp", "")

                # Map each page to all environments it appeared in
                page_to_envs = defaultdict(set)
                all_envs = set()

                # Expand summary rows to include env context
                for r in data.get("results", []):
                    page_to_envs[r.get("page")].add(r.get("env"))
                    all_envs.add(r.get("env"))

                # Expand summary rows to include env context
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

                # Make sure there is no missing envs
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

    df = pd.DataFrame(records)
    return df
