
"""
BaseController

This is the shared skeleton for all agent controllers in the framework.
Every agent follows the same basic lifecycle: setup, run, teardown. This keeps everything consistent so the
orchestrator doesn't have to treat each agent differently.

Each subclass fills in the agent specific details, but all the shared logic
(storing results, running) lives here.
"""


from abc import ABC, abstractmethod
import asyncio


class BaseController(ABC):
    # Subclasses should set these so we know which agent/model produced the results.
    AGENT_NAME: str = None
    MODEL_NAME: str = None

    def __init__(self, model, env, page, task):

        # Store the basic context for this agent run.
        #- model: which LLM or backend the agent is using
        #- page: which test page (AIA.html, relaxedEIA, strictEIA, etc.)
        #- task: the specific instruction for this run

        self.model = model
        self.env = env
        self.page = page
        self.task = task

        # This will hold whatever results/metrics the agent produces.
        self.results = []

        if not getattr(self, "AGENT_NAME", None) or not getattr(self, "MODEL_NAME", None):
            raise NotImplementedError(
                f"{self.__class__.__name__} must define AGENT_NAME and MODEL_NAME constants."
            )


    # Anything the agent needs before running.
    # Examples:
    # - launching a browser
    # - spinning up an agent
    # - initializing a grounding model
    # This depends entirely on the subclass.

    @abstractmethod
    async def setup(self):
        pass


    # The actual work the agent performs.
    #
    # Whatever metric this returns is what gets logged. Different agents have
    # slightly different behaviors, but they all eventually produce some kind
    # of success/failure result that the Dashboard will display.

    @abstractmethod
    async def run(self):
        pass


    # Clean up anything the agent created.
    #
    # This could be closing, Playwright, shutting down processes, etc.
    # Basically the opposite of setup().

    @abstractmethod
    async def teardown(self):
        pass

    #
    # Store a metric result in a consistent structure.
    #
    # All agents log results the same way, which makes the Dashboard and
    # analytics side a lot easier. If a metric comes back as a dict, we
    # extract the 'result' field — otherwise we store it directly.

    async def record(self, metric):
        metric_val = metric.get("result") if isinstance(metric, dict) else metric
        self.results.append({
            "model": self.model,
            "env": self.env,
            "page": self.page,
            "task": self.task,
            "metric": metric_val
        })
        return self.results





    # Run the full lifecycle:
    # setup, run, record, teardown
    #
    # This is what the orchestrator calls.

    async def execute(self):
        await self.setup()
        metric = await self.run()
        results = await self.record(metric)
        await self.teardown()
        return results