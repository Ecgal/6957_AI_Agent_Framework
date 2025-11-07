# Controllers/LiteWebAgent_Controller.py
import asyncio
import json
import os
import sys
import tempfile
import textwrap
from Controllers.BaseController import BaseController
from dotenv import load_dotenv
from utils import load_environment, wait_for_metric

AGENT_NAME = "LiteWebAgent"
MODEL_NAME = "gpt-4o"


class LiteWebAgentController(BaseController):
    AGENT_NAME = AGENT_NAME
    MODEL_NAME = MODEL_NAME

    def __init__(self, model, env, page, task):
        super().__init__(model, env, page, task)
        load_environment()
        self.website = f"http://localhost:8080/environments/{env}/{page}"

    async def setup(self):
        print(f" Setting up LiteWebAgent for {self.website}")

    async def run(self):
        try:
            #  Create an isolated runner
            runner_code = textwrap.dedent(
                r"""
                import os, sys, json, asyncio, datetime, traceback
                from dotenv import load_dotenv
                from utils import wait_for_metric

                # --- Setup logging ---
                log_path = os.path.join(os.getcwd(), f"litewebagent_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                sys.stdout = open(log_path, "w", encoding="utf-8")
                sys.stderr = sys.stdout
                print(f"[LWA] Logging to {log_path}")

                # --- Setup paths ---
                def prepare_paths():
                    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    LITE_PATH = os.path.join(ROOT, "LiteWebAgent")
                    sys.path.insert(0, ROOT)
                    sys.path.insert(0, LITE_PATH)
                    os.chdir(ROOT)
                    print(f"[LWA] ROOT: {ROOT}")
                    print(f"[LWA] LITE_PATH: {LITE_PATH}")
                    return ROOT, LITE_PATH

                # --- Load environment variables ---
                def setup_env(ROOT):
                    dotenv_path = os.path.join(ROOT, ".env")
                    if os.path.exists(dotenv_path):
                        load_dotenv(dotenv_path)
                        print("[LWA]  Loaded .env successfully")
                    else:
                        print("[LWA] ️ No .env file found")

                # --- Initialize Playwright Manager ---
                def init_playwright():
                    from LiteWebAgent.litewebagent.webagent_utils_sync.utils import playwright_manager
                    orig_setup = playwright_manager.setup_playwright

                    def patched_setup_playwright(*args, **kwargs):
                        kwargs.pop("storage_state", None)
                        kwargs["storage_state"] = None
                        return orig_setup(*args, **kwargs)

                    playwright_manager.setup_playwright = patched_setup_playwright
                    print("[LWA]  Playwright manager patched successfully")

                    pm = playwright_manager.setup_playwright(headless=False)
                    print(f"[LWA]  Playwright initialized: {pm}")
                    return pm

                # --- Create the LiteWebAgent ---
                def create_agent(start_url, goal, model_name, pm):
                    from LiteWebAgent.litewebagent.core import agent_factory
                    print("[LWA]  Creating web agent...")
                    agent = agent_factory.setup_prompting_web_agent(
                        starting_url=start_url,
                        goal=goal,
                        playwright_manager=pm,
                        model_name=model_name,
                        agent_type="PromptAgent",
                        log_folder="log",
                        headless=False,
                        storage_state=None
                    )
                    print("[LWA]  Agent created successfully")
                    return agent

                # --- Run task and wait for metric ---
                def run_task(agent, goal):
                    print(f"[LWA]  Running prompt: {goal}")
                    try:
                        reply = agent.send_prompt(goal)
                        print("[LWA]  Waiting for metric signal...")

                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        done = loop.run_until_complete(wait_for_metric(timeout=90))
                        loop.close()

                        if done and done.get("result") != "timeout":
                            print(f"[LWA]  Metric received: {done['result']}")
                            result = done["result"]
                        else:
                            print("[LWA] ️ No metric received (timeout).")
                            result = "timeout"

                        print(f"[LWA]  Model replied: {str(reply)[:400]}")
                        return {"ok": True, "result": result}

                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"[LWA]  Task failed: {e}\n--- TRACEBACK ---\n{tb}")
                        return {"ok": False, "error": str(e), "traceback": tb}

                # --- Main entry ---
                def main():
                    start_url, goal, model_name = sys.argv[1:4]
                    ROOT, _ = prepare_paths()
                    setup_env(ROOT)

                    pm = None
                    try:
                        pm = init_playwright()
                        agent = create_agent(start_url, goal, model_name, pm)
                        result = run_task(agent, goal)
                    finally:
                        # --- Always teardown browser ---
                        try:
                            if pm:
                                print("[LWA]  Closing Playwright browser...")
                                pm.close_all()
                                print("[LWA]  Browser closed cleanly.")
                        except Exception as e:
                            print(f"[LWA]  Error during teardown: {e}")

                    print(json.dumps(result))

                if __name__ == "__main__":
                    try:
                        main()
                    except Exception as e:
                        print(json.dumps({"ok": False, "error": str(e)}))
                        sys.exit(1)
                """
            )

            #  Write runner file
            with tempfile.TemporaryDirectory() as td:
                runner_path = os.path.join(td, "litewebagent_runner.py")
                with open(runner_path, "w", encoding="utf-8") as f:
                    f.write(runner_code)

                proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                pyenv = os.environ.copy()
                extra_paths = [proj_root, os.path.join(proj_root, "LiteWebAgent")]
                pyenv["PYTHONPATH"] = os.pathsep.join(extra_paths + [pyenv.get("PYTHONPATH", "")])

                #  launch subprocess for LiteWebAgent
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    runner_path,
                    self.website,
                    self.task,
                    self.model,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=pyenv,
                    cwd=proj_root,
                )

                # start watching for metric concurrently
                metric_future = asyncio.create_task(wait_for_metric(timeout=120))
                proc_future = asyncio.create_task(proc.wait())

                done, _ = await asyncio.wait(
                    {metric_future, proc_future},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                #  if metric arrived first, kill the process
                if metric_future in done:
                    metric_data = metric_future.result()
                    metric = metric_data.get("result", "timeout")
                    print(f" Metric {metric} received — killing LiteWebAgent.")
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        proc.kill()
                    return {"result": metric}

                #otherwise process end naturally
                stdout, stderr = await proc.communicate()
                out = stdout.decode(errors="ignore").strip()
                err = stderr.decode(errors="ignore").strip()

                try:
                    payload = json.loads(out.splitlines()[-1])
                    return {"result": payload.get("result", "completed")}
                except Exception:
                    return {"result": f"error: could not parse subprocess output: {out or err}"}

        except Exception as e:
             return {"result": f"error: {e}"}

    async def teardown(self):
        print(f" Tearing down LiteWebAgent for {self.env}/{self.page}")
        await asyncio.sleep(0)


# discovery point
async def run(prompts_file="Prompts/prompts.json"):
    with open(prompts_file) as f:
        tasks = json.load(f)

    results = []
    for t in tasks:
        ctrl = LiteWebAgentController(
            model=MODEL_NAME,
            env=t["env"],
            page=t["page"],
            task=t["task"],
        )
        res = await ctrl.run()
        results.append(res)
    return results
