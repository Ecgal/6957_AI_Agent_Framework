import asyncio
import subprocess
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()


#permissions are pretty open to communicate to our metrics server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# store the metrics received, for example, after one log it should look like {"last":{"result":"AIA}}
metrics_results = {}



# This is to handle the CORS preflight OPTIONS request before POST
@app.options("/log")
async def options_log():
    response = JSONResponse(content={"status": "ok"})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# receives the metrics in JSON and saves into Metrics_results
@app.post("/log")
async def log_event(request: Request):
    data = await request.json()
    result = data.get("result")

    print(f" Metric logged: {result}")
    metrics_results["last"] = {"result": result}
    return {"status": "ok"}


#helper to actively wait for the metrics
async def wait_for_metric(timeout):
    start = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start) < timeout:
        if metrics_results:
            key, value = metrics_results.popitem()
            return {"result": value}
        await asyncio.sleep(0.5)
    return {"result": "timeout"}

#start the FastAPI server ( where our metrics log is)
def run_metrics_server(host="localhost", port=5000):
    uvicorn.run(app, host=host, port=port)

# Web server for our environment
def start_web_server(port=8080):
    return subprocess.Popen(["python3", "-m", "http.server", str(port)])

#Stop the webserver used after the tests  are done
def stop_web_server(proc):
    proc.terminate()


#takes all the raw results from then group by page.
def summarize_results(results):
    #used to create a dictionary where each key is a page name
    counts = defaultdict(lambda: {"success": 0, "total": 0})

    #  count the success based on all results. example: {AIA.html":{success" 1, "total": 2}}
    for res in results:
        page = res["page"]
        metric = res["metric"]

        counts[page]["total"] += 1
        if metric != "finished":
            counts[page]["success"] += 1

    # build the final summary table
    summary = []
    for page, vals in counts.items():
        success_rate = vals["success"] / vals["total"] * 100
        summary.append({
            "page": page,
            "success_rate": round(success_rate, 1), #round to the first decimal place
            "successes": vals["success"],
            "total": vals["total"]
        })
    return summary