import json
from datetime import datetime
from pathlib import Path

def log_summary_to_json(summary, results, agent_name=None, model_name=None, filename=None, results_dir="results"):
    Path(results_dir).mkdir(exist_ok=True)

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