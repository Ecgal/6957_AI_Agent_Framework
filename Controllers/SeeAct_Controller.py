import os
import json
import asyncio
from dotenv import load_dotenv
from seeact.agent import SeeActAgent
from utils import wait_for_metric


# SeeAct agent controller
load_dotenv()
api_key = os.getenv("API_KEY")

async def run_seeact(prompts_file="prompts.json"):

    #load our task definitions  JSON file
    with open(prompts_file) as f:
        tasks = json.load(f)

    results = []

    #itterate through each task in the JSON file
    for task_obj in tasks:
        env = task_obj["env"]
        page = task_obj["page"]
        task = task_obj["task"]


        website = f"http://localhost:8080/environments/{env}/{page}"
        print(f" Running SeeAct on {website}")

        #SeeAct info, will need to switch the model in the future
        agent = SeeActAgent(
            model="gpt-4o",
            default_website=website,
            default_task=task
        )

        try:
            await agent.start()
            metric_result = None

            #SeeAct running logic, main loop waits for agent task to complete or metric received
            while not agent.complete_flag and metric_result is None:
                prediction_dict = await agent.predict()
                await agent.execute(prediction_dict)

                # check for metric after each SeeAct action
                done = await wait_for_metric(timeout=1)  #in seconds

                #metric received
                if done and done.get("result") != "timeout":
                    #wait for metric uses a dictionary {"result":"AIA"} so ave the result to "result"
                    metric_result = done["result"]
                    print(f" Metric received: {metric_result}")
                    break

            # if no metric ever arrived during the loop, do one last wait
            if metric_result is None:
                done = await wait_for_metric(timeout=30) #in seconds
                metric_result = done.get("result", "timeout")


            print(f"{env}/{page} finished with: {metric_result}")

            results.append({
                "env": env,
                "page": page,
                "task": task,
                "metric": str(metric_result)
            })

        #catch any errors and append the results to reflect the error
        except Exception as e:
            print(f" Error during {env}/{page}: {e}")
            results.append({
                "env": env,
                "page": page,
                "task": task,
                "metric": f"error: {e}"
            })

        finally:
            await agent.stop()

    return results

if __name__ == "__main__":
    asyncio.run(run_seeact())