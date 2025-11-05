import os
import asyncio
import threading
import importlib
import inspect
import uvloop

import sys, os
from logger import log_summary_to_json
from utils import start_web_server, stop_web_server, run_metrics_server, summarize_results

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "LiteWebAgent"))

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


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
        print(f" {controller_name} has no run() or run_seeact() method.")
        return []


async def main():
    env_server = None
    all_results = []

    try:
        print("Starting environment server on port 8080...")
        env_server = start_web_server(port=8080)

        print(" Starting metrics logging server on port 5050...")
        metrics_thread = threading.Thread(
            target=run_metrics_server,
            kwargs={"host": "127.0.0.1", "port": 5050},
            daemon=True
        )
        metrics_thread.start()

        await asyncio.sleep(2)

        print("Discovering controllers...")
        controllers = await discover_controllers()

        if not controllers:
            print("No valid controllers found in /Controllers.")
            return

        for controller_name, module_path, agent_name, model_name in controllers:
            print(f"\n Running {agent_name} ({controller_name}) using {model_name}...")
            results = await run_controller(controller_name, module_path, "Prompts/prompts.json")

            # Append all results for summary
            all_results.extend(results)

            print(f"\n Completed {agent_name}")
            for res in results:
                print(f"{res['env']} | {res['page']} | {res['task']} | {res['metric']}")

            summary = summarize_results(results)
            log_summary_to_json(
                summary=summary,
                results=results,
                agent_name=agent_name,
                model_name=model_name
            )

        print("\n---ALL CONTROLLERS COMPLETE---")
        final_summary = summarize_results(all_results)
        for s in final_summary:
            print(f"{s['page']}: {s['success_rate']}% success ({s['successes']}/{s['total']})")

    except Exception as e:
        print(f" Error during test run: {e}")

    finally:
        print("\n Stopping servers...")
        stop_web_server(env_server)
        print(" Done.")


if __name__ == "__main__":
    asyncio.run(main())

