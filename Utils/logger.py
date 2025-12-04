

"""
This helper writes out the full results of a test run into a JSON file.
Each run gets its own timestamped results file so we can:
- Track multiple runs over time
- Feed the dashboard clean, structured data
- Store both raw per page results and the summarized metrics

"""



import json
from datetime import datetime
from pathlib import Path


def log_summary_to_json(summary, results, agent_name=None, model_name=None, filename=None, results_dir="results"):
    # Make sure the results directory exists
    Path(results_dir).mkdir(exist_ok=True)

    # Auto generate a timestamped filename if none is provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"results_{timestamp}.json"

    filepath = Path(results_dir) / filename

    data = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name or "unknown_agent",
        "model": model_name or "unknown_model",
        "summary": summary,
        "results": results
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote full run data to : {filepath}")