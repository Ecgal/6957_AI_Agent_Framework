import asyncio
import threading
import importlib
import uvloop

from logger import log_summary_to_json
from utils import start_web_server, stop_web_server, run_metrics_server, summarize_results

# use uvloop for async
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def run_controller(controller_name, prompts_file):
    CONTROLLERS = {
        "seeact": "Controllers.SeeAct_Controller",
    }

    module_path = CONTROLLERS.get(controller_name.lower())
    if not module_path:
        raise ValueError(f"Unknown controller '{controller_name}'")

    module = importlib.import_module(module_path)

    if hasattr(module, "run"):
        return await module.run(prompts_file)
    elif hasattr(module, "run_seeact"):
        return await module.run_seeact(prompts_file)
    else:
        raise RuntimeError(f"Controller {controller_name} does not have a 'run()' or 'run_seeact()' method.")

async def main():
    env_server = None
    try:
        print("Starting environment server on port 8080...")
        env_server = start_web_server(port=8080)

        print("Starting metrics logging server on port 5050...")
        metrics_thread = threading.Thread(
            target=run_metrics_server,
            kwargs={"host": "127.0.0.1", "port": 5050},
            daemon=True
        )
        metrics_thread.start()

        await asyncio.sleep(2)  # give servers time to start

        print(" Running SeeAct tests...")
        seeact_results = await run_controller("seeact", "Prompts/prompts.json")

        print("\n TEST SUMMARY")
        for res in seeact_results:
            print(f"{res['env']} | {res['page']} | {res['task']} | {res['metric']}")

        summary = summarize_results(seeact_results)

        print("\n SUCCESS RATES")
        for s in summary:
            print(f"{s['page']}: {s['success_rate']}% success ({s['successes']}/{s['total']})")

        log_summary_to_json(
            summary=summary,
            results=seeact_results,
            agent_name="SeeAct",
            model_name="gpt-4o"
        )

    except Exception as e:
        print(f" Error during test run: {e}")

    finally:
        print("\nStopping servers...")
        stop_web_server(env_server)
        print(" Done.")

if __name__ == "__main__":
    asyncio.run(main())
