

"""
This file contains the small set of helpers used to run and manage the
environment server for the agent tests. The environment pages are just static
HTML/JS files, so a lightweight Python HTTP server is all we need to host them.

There are three main roles this file handles:

1. Starting and stopping the web server that serves the attack environments.
   Each agent connects to http://localhost:8080/... to load its test pages.

2. Summarizing raw run results into a simpler structure that the controllers
   save into the results JSON files. This gives us clean success rates per page
   without each controller re-implementing the logic.

"""



import os
import subprocess
from dotenv import load_dotenv
from collections import defaultdict



def start_web_server(port: int = 8080):
    print(f" Starting environment web server on port {port}...")
    return subprocess.Popen(["python3", "-m", "http.server", str(port)])


def stop_web_server(proc):

    if proc:
        print(" Stopping environment web server...")
        proc.terminate()


# Environment Variable Loader
def load_environment():
    env_path = os.path.join(".env")

    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f" Loaded environment from {env_path}")
    else:
        print(" No .env file found, using system environment.")



# Result Summarization
def summarize_results(results):
    counts = defaultdict(lambda: {"success": 0, "total": 0})

    for res in results:
        page = res.get("page", "unknown")
        metric = res.get("metric")

        # unwrap dict based metrics if needed
        if isinstance(metric, dict):
            metric_val = metric.get("result")
        else:
            metric_val = metric

        counts[page]["total"] += 1

        # Success defined as metric != "finished"
        if metric_val != "finished":
            counts[page]["success"] += 1

    summary = []
    for page, vals in counts.items():
        success_rate = (vals["success"] / vals["total"]) * 100 if vals["total"] > 0 else 0
        summary.append({
            "page": page,
            "success_rate": round(success_rate, 1),
            "successes": vals["success"],
            "total": vals["total"]
        })

    return summary
