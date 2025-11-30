# metrics_server.py
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Metrics Server")

# Allow broad access for agent communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_results = {}

# Handle preflight CORS (OPTIONS)
@app.options("/log")
async def options_log():
    response = JSONResponse(content={"status": "ok"})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# Log endpoint — receives metrics from agents
@app.post("/log")
async def log_event(request: Request):
    data = await request.json()
    result = data.get("result")
    print(f" Metric logged: {result}")
    metrics_results["last"] = {"result": result}
    return {"status": "ok"}


# Wait for metric (used by controllers to poll)
async def wait_for_metric(timeout: float = 10.0):
    start = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start) < timeout:
        if metrics_results:
            key, value = metrics_results.popitem()
            return {"result": value}
        await asyncio.sleep(0.5)
    return {"result": "timeout"}


# Run the metrics FastAPI server (blocking call)
def run_metrics_server(host: str = "127.0.0.1", port: int = 5050):
    print(f" Starting metrics server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

