"""
This file is the brain of the entire framework. Its job is to coordinate
every moving part involved in running attack evaluations across different
agents, models, and environments.

At a high level, the orchestrator is responsible for:

1. Starting and managing all required servers:
   - The static environment web server (serving the environment + attack pages)
   - The metrics server (FastAPI) that receives success/failure signals

2. Initializing the ModelManager singleton
   This ensures that GPT or MiniCPM is loaded once and shared across all
   controllers instead of every controller loading its own model.

3. Discovering all controllers dynamically
   Any controller placed in the /Controllers folder will automatically be
   detected and executed without hardcoding anything. This keeps the system
   extensible as new agents are added.

4. Running each controller in sequence
   For each agent:
     - Load its prompts
     - Execute its run() method
     - Collect task results
     - Summarize and log output to a JSON results file

   Multiple iterations can be run back-to-back for benchmarking.

5. Handling global test lifecycle
   - Ensures servers stay alive throughout runs
   - Collects all results into a final summary at the end
   - Cleans up servers on shutdown

"""

import os
import sys
import asyncio
import threading
import importlib
import inspect
import uvloop

from Utils.logger import log_summary_to_json
 # from utils import start_web_server, stop_web_server, run_metrics_server, summarize_results
from Utils.model_manager import ModelManager
from Servers.metrics_server import run_metrics_server
from Servers.environment_server import start_web_server, stop_web_server, summarize_results

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "LiteWebAgent"))

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Use GPT-4o
# python orchestrator.py gpt

# Use MiniCPM local model
# python orchestrator.py minicpm


async def discover_controllers():
    controllers_dir = os.path.join(os.path.dirname(__file__), "Controllers")
    controllers = []

    for filename in os.listdir(controllers_dir):
        if not filename.endswith(".py") or filename == "BaseController.py":
            continue

        controller_name = filename.replace(".py", "")
        module_path = f"Controllers.{controller_name}"
        module = importlib.import_module(module_path)

        agent_name = getattr(module, "AGENT_NAME", controller_name)
        model_name = getattr(module, "MODEL_NAME", "unknown")

        if hasattr(module, "run") and inspect.iscoroutinefunction(module.run):
            controllers.append((controller_name, module_path, agent_name, model_name))
        elif hasattr(module, "run_seeact") and inspect.iscoroutinefunction(module.run_seeact):
            controllers.append((controller_name, module_path, agent_name, model_name))

    return controllers


async def run_controller(controller_name, module_path, prompts_file):
    module = importlib.import_module(module_path)

    if hasattr(module, "run"):
        return await module.run(prompts_file)
    elif hasattr(module, "run_seeact"):
        return await module.run_seeact(prompts_file)
    else:
        print(f" {controller_name} has no run() method.")
        return []


async def main():
    # Parse command line arguments for model type
    model_type = sys.argv[1] if len(sys.argv) > 1 else "gpt"

    # Parse command line arguments for model type
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Initialize ModelManager singleton before starting tests
    print(f"\n Initializing ModelManager with model_type: {model_type}")
    try:
        model_manager = ModelManager.get(model_type=model_type)
        print(f" ModelManager initialized successfully")
    except Exception as e:
        print(f" Failed to initialize ModelManager: {e}")
        return

    env_server = None
    all_results = []

    try:
        print("\n Starting environment server on port 8080...")
        env_server = start_web_server(port=8080)

        print(" Starting metrics logging server on port 5050...")
        metrics_thread = threading.Thread(
            target=run_metrics_server,
            kwargs={"host": "127.0.0.1", "port": 5050},
            daemon=True
        )
        metrics_thread.start()

        await asyncio.sleep(2)

        print("\n Discovering controllers...")
        controllers = await discover_controllers()

        if not controllers:
            print(" No valid controllers found in /Controllers.")
            return

        for iteration in range(1, iterations + 1):
            print(f"\n==============================")
            print(f" STARTING ITERATION {iteration}/{iterations}")
            print(f"==============================")
            for controller_name, module_path, agent_name, model_name in controllers:
                print(f"\n Running {agent_name} ({controller_name}) using {model_type}...")
                results = await run_controller(controller_name, module_path, "Prompts/prompts.json")

                # Append all results for summary
                all_results.extend(results)

                print(f"\n Completed {agent_name}")
                for res in results:
                    print(f"  {res['env']} | {res['page']} | {res['task']} | {res['metric']}")

                summary = summarize_results(results)
                log_summary_to_json(
                    summary=summary,
                    results=results,
                    agent_name=agent_name,
                    model_name=model_type  # Log actual model type used
                )


        print("\n---ALL ITERATIONS COMPLETE---")
        final_summary = summarize_results(all_results)
        for s in final_summary:
            print(f"  {s['page']}: {s['success_rate']}% success ({s['successes']}/{s['total']})")

    except Exception as e:
        print(f" Error during test run: {e}")

    finally:
        print("\n Stopping Servers...")
        stop_web_server(env_server)
        print(" Done.")


if __name__ == "__main__":
    asyncio.run(main())