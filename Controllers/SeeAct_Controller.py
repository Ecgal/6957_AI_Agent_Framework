import os
import json
import asyncio
from dotenv import load_dotenv
from seeact.agent import SeeActAgent
from utils import wait_for_metric
from model_manager import ModelManager

AGENT_NAME = "SeeAct"
MODEL_NAME = "dynamic"  # Will be set based on ModelManager

# SeeAct-specific ranking model path
RANKER_PATH = "/uufs/chpc.utah.edu/common/home/u1533682/SeeAct/SeeAct/seeact_package/seeact/model/deberta-v3-base"

load_dotenv()


async def run_seeact(prompts_file="prompts.json"):
    # Get the global ModelManager instance
    model_manager = ModelManager.get()

    # Pre-load ranking model if using MiniCPM (since SeeAct always needs it)
    if model_manager.model_type == "minicpm":
        # Ensure ranking model is loaded before running tasks
        model_manager.get_ranking_model(ranker_path=RANKER_PATH)

    # Load task definitions from JSON file
    with open(prompts_file) as f:
        tasks = json.load(f)

    results = []

    # Iterate through each task in the JSON file
    for task_obj in tasks:
        env = task_obj["env"]
        page = task_obj["page"]
        task = task_obj["task"]

        website = f"http://localhost:8080/environments/{env}/{page}"
        print(f"\n🌐 Running SeeAct on {website}")

        # Initialize agent based on model type
        if model_manager.model_type == "gpt":
            agent = SeeActAgent(
                model="gpt-4o",
                default_website=website,
                default_task=task
            )

        elif model_manager.model_type == "minicpm":
            agent = SeeActAgent(
                model="minicpm",
                ranker_path=RANKER_PATH,
                default_website=website,
                default_task=task
            )

        try:
            await agent.start()
            metric_result = None

            # SeeAct main loop: wait for agent task completion or metric received
            while not agent.complete_flag and metric_result is None:
                prediction_dict = await agent.predict()
                await agent.execute(prediction_dict)

                # Check for metric after each SeeAct action
                done = await wait_for_metric(timeout=1)  # in seconds

                # Metric received
                if done and done.get("result") != "timeout":
                    metric_result = done["result"]
                    print(f"✅ Metric received: {metric_result}")
                    break

            # If no metric arrived during the loop, do one last wait
            if metric_result is None:
                done = await wait_for_metric(timeout=30)  # in seconds
                metric_result = done.get("result", "timeout")

            print(f"📊 {env}/{page} finished with: {metric_result}")

            results.append({
                "env": env,
                "page": page,
                "task": task,
                "metric": metric_result if isinstance(metric_result, str) else metric_result.get("result")
            })

        # Catch any errors and append the results to reflect the error
        except Exception as e:
            print(f"❌ Error during {env}/{page}: {e}")
            results.append({
                "env": env,
                "page": page,
                "task": task,
                "metric": f"error: {e}"
            })

        finally:
            await agent.stop()

    return results


async def run(prompts_file="Prompts/prompts.json"):
    return await run_seeact(prompts_file)


if __name__ == "__main__":
    asyncio.run(run_seeact())