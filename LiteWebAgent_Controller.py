# Controllers/LiteWebAgent_Controller.py
import asyncio
import json
import os
import sys
import tempfile
import textwrap
from Controllers.BaseController import BaseController
from utils import load_environment

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
        print(f"🧭 Setting up LiteWebAgent for {self.website}")

    async def run(self):
        """
        Run LiteWebAgent in an ISOLATED SUBPROCESS so Playwright Sync API
        never sees the parent asyncio loop.
        """
        try:
            # Build the isolated runner
            runner_code = textwrap.dedent(
                r"""
                import os, sys, json
                ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                sys.path.append(ROOT)
                sys.path.append(os.path.join(ROOT, "LiteWebAgent"))

                from LiteWebAgent.litewebagent.webagent_utils_sync.utils import playwright_manager
                from LiteWebAgent.litewebagent.core import agent_factory

                def main():
                    start_url = sys.argv[1]
                    goal = sys.argv[2]
                    model_name = sys.argv[3]

                    # --- Patch Playwright manager to ignore 'state.json' ---
                    orig_setup = playwright_manager.setup_playwright
                    def patched_setup_playwright(*args, **kwargs):
                        kwargs.pop("storage_state", None)
                        kwargs["storage_state"] = None  # ensures no state.json file used
                        return orig_setup(*args, **kwargs)
                    playwright_manager.setup_playwright = patched_setup_playwright

                    # --- Continue normal setup ---
                    pm = playwright_manager.setup_playwright(headless=True)
                    agent = agent_factory.setup_prompting_web_agent(
                        starting_url=start_url,
                        goal=goal,
                        playwright_manager=pm,
                        model_name=model_name,
                        agent_type="PromptAgent",
                        log_folder="log",
                        headless=True,
                        storage_state=None  # also redundant safeguard
                    )

                    try:
                        print(f"🤖 Running LiteWebAgent single-step task: {goal}")
                        reply = agent.send_prompt(goal)  # ✅ no kwargs
                        if reply is None:
                            raise RuntimeError("send_prompt() returned None")
                        print(f"💬 Model replied: {str(reply)[:500]}")
                        result = "completed"
                    except Exception as e:
                        raise RuntimeError(f"Failed to execute prompt: {e}")
                    
                    print(json.dumps({"ok": True, "result": result}))

                if __name__ == "__main__":
                    try:
                        main()
                    except Exception as e:
                        print(json.dumps({"ok": False, "error": str(e)}))
                        sys.exit(1)
                """
            )

            # Write runner to a temporary file
            with tempfile.TemporaryDirectory() as td:
                runner_path = os.path.join(td, "lwa_runner.py")
                with open(runner_path, "w", encoding="utf-8") as f:
                    f.write(runner_code)

                # Set up proper PYTHONPATH for the subprocess
                proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                pyenv = os.environ.copy()
                extra_paths = [proj_root, os.path.join(proj_root, "LiteWebAgent")]
                pyenv["PYTHONPATH"] = os.pathsep.join(extra_paths + [pyenv.get("PYTHONPATH", "")])

                # Launch the subprocess
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    runner_path,
                    self.website,
                    self.task,
                    self.model,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=pyenv,
                )
                stdout, stderr = await proc.communicate()

            out = stdout.decode("utf-8", errors="ignore").strip()
            err = stderr.decode("utf-8", errors="ignore").strip()

            if not out:
                if err:
                    print(f"❌ LiteWebAgent stderr:\n{err}")
                return {"result": f"error: empty output from subprocess"}

            try:
                payload = json.loads(out.splitlines()[-1])
            except Exception:
                return {"result": f"error: could not parse subprocess output: {out or err}"}

            if not payload.get("ok"):
                return {"result": f"error: {payload.get('error', 'unknown error')}"}

            return {"result": payload.get("result", 'completed')}

        except Exception as e:
            return {"result": f"error: {e}"}


# --- Discovery entrypoint for Testing_Controller.py ---
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
        res = await ctrl.execute()
        results.extend(res)
    return results
