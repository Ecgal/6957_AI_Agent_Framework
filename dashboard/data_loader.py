from collections import defaultdict
import json, glob
import pandas as pd

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

                page_to_envs = defaultdict(set)
                all_envs = set()
                for r in data.get("results", []):
                    page_to_envs[r.get("page")].add(r.get("env"))
                    all_envs.add(r.get("env"))

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
