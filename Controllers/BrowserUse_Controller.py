import os
import json
import asyncio
from dotenv import load_dotenv
from browser_use import Agent, Browser, ChatOpenAI, ChatMiniCPM
from utils import wait_for_metric
from model_manager import ModelManager

AGENT_NAME = "BrowserUse"
MODEL_NAME = "dynamic"  # Will be set based on ModelManager

load_dotenv()

async def run_browseruse(prompts_file="Prompts/prompts.json"):
    # Get the global ModelManager instance
    model_manager = ModelManager.get()

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

            # Initialize agent based on model type
            if model_manager.model_type == "gpt":
                llm = ChatOpenAI(model="gpt-4o")
                agent = Agent(task=full_task, llm=llm, browser=browser)

            elif model_manager.model_type == "minicpm":
                llm = ChatMiniCPM()
                agent = Agent(
                    task=full_task,
                    directly_open_url=True,
                    use_vision=True,
                    llm=llm,
                    initial_actions=[
                        {"navigate": {"url": website, "new_tab": False}}
                    ],
                    browser=browser,
                )

            print(f" Starting BrowserUse agent with {model_manager.model_type}...")

            # Run agent task asynchronously
            agent_task = asyncio.create_task(agent.run())

            metric_result = None

            # Wait for either agent completion or metric arrival
            while not agent_task.done() and metric_result is None:
                done = await wait_for_metric(timeout=1)
                if done and done.get("result") != "timeout":
                    metric_result = done["result"]
                    print(f" Metric received: {metric_result}")
                    break
                await asyncio.sleep(1)

            # If metric received, stop agent
            if metric_result:
                if not agent_task.done():
                    agent_task.cancel()
                    print(" Stopping BrowserUse (metric received).")
            else:
                metric_result = "timeout"
                print(" No metric received — marking as timeout.")
                if not agent_task.done():
                    agent_task.cancel()

            print(f" {env}/{page} finished with: {metric_result}")

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

            for task in asyncio.all_tasks():
                if task is not asyncio.current_task() and "browser_use" in str(task):
                    task.cancel()

    return results


async def run(prompts_file="Prompts/prompts.json"):
    return await run_browseruse(prompts_file)


if __name__ == "__main__":
    asyncio.run(run_browseruse())