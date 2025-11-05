import os
import json
import asyncio
from dotenv import load_dotenv
from browser_use import Agent, Browser, ChatOpenAI
from utils import wait_for_metric

AGENT_NAME = "BrowserUse"
MODEL_NAME = "gpt-4o"

load_dotenv()


async def run_browseruse(prompts_file="Prompts/prompts.json"):
    with open(prompts_file) as f:
        tasks = json.load(f)

    results = []

    for task_obj in tasks:
        env = task_obj["env"]
        page = task_obj["page"]
        task = task_obj["task"]

        website = f"http://localhost:8080/environments/{env}/{page}"
        print(f"\n Running BrowserUse on {website}")

        full_task = (
            f"Open this webpage: {website}\n\n"
            f"Then complete the following task:\n{task}"
        )

        browser = None
        agent = None

        try:

            browser = Browser(headless=False)
            llm = ChatOpenAI(model=MODEL_NAME)
            agent = Agent(task=full_task, llm=llm, browser=browser)

            print(" Starting BrowserUse agent...")


            agent_task = asyncio.create_task(agent.run())

            metric_result = None


            while not agent_task.done() and metric_result is None:
                done = await wait_for_metric(timeout=1)
                if done and done.get("result") != "timeout":
                    metric_result = done["result"]
                    print(f" Metric received: {metric_result}")
                    break
                await asyncio.sleep(1)


            if metric_result:
                if not agent_task.done():
                    agent_task.cancel()
                    print(" Stopping BrowserUse (metric received).")
            else:

                metric_result = "timeout"
                print(" No metric received — marking as timeout.")
                if not agent_task.done():
                    agent_task.cancel()


            print(f"{env}/{page} finished with: {metric_result}")

            results.append({
                "env": env,
                "page": page,
                "task": task,
                "metric": metric_result,
            })

        except Exception as e:
            print(f" Error during {env}/{page}: {e}")
            results.append({
                "env": env,
                "page": page,
                "task": task,
                "metric": f"error: {e}",
            })

        finally:
            # Graceful shutdown of BrowserUse and Chrome process
            try:
                if agent and hasattr(agent, "close"):
                    await agent.close()
            except Exception:
                pass

            try:
                if browser:
                    await browser.close()
            except Exception:
                pass

            # Cleanup stray asyncio tasks created by browser_use
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task() and "browser_use" in str(task):
                    task.cancel()

    return results

async def run(prompts_file="Prompts/prompts.json"):
    return await run_browseruse(prompts_file)


if __name__ == "__main__":
    asyncio.run(run_browseruse())
