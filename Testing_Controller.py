import asyncio
import subprocess
import sys
import threading
import uvloop

from utils import start_web_server, stop_web_server, run_metrics_server, summarize_results
from Controllers.SeeAct_Controller import run_seeact

#use uvloop for async
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def main():

    env_server = None
    try:
        # Start environment server
        print("Starting environment server on port 8080...")
        env_server = start_web_server(port=8080)

        # Start metrics logging server and run in the background
        print("Starting metrics logging server on port 5050...")
        metrics_thread = threading.Thread(  #create new thread
            target=run_metrics_server,
            kwargs={"host": "127.0.0.1", "port": 5050},
            daemon=True
        )
        metrics_thread.start()

        await asyncio.sleep(2)  # give servers a moment to start

        #  Run model controllers, currently just seeAct
        print(" Running SeeAct tests...")
        seeact_results = await run_seeact(prompts_file="Prompts/prompts.json")


        # Print and summerize the results
        print("\n TEST SUMMARY")
        for res in seeact_results:
            print(f"{res['env']} | {res['page']} | {res['task']} | {res['metric']}")

        print("\n SUCCESS RATES")
        for s in summarize_results(seeact_results):
            print(f"{s['page']}: {s['success_rate']}% success ({s['successes']}/{s['total']})")

    except Exception as e:
        print(f" Error during test run: {e}")

    finally:
        # stop the servers
        print("\nStopping servers...")
        stop_web_server(env_server)
        print(" Done.")

if __name__ == "__main__":
    asyncio.run(main())