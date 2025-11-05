from abc import ABC, abstractmethod
import asyncio

class BaseController(ABC):
    AGENT_NAME: str = None
    MODEL_NAME: str = None

    def __init__(self, model, env, page, task):
        self.model = model
        self.env = env
        self.page = page
        self.task = task
        self.results = []

        if not getattr(self, "AGENT_NAME", None) or not getattr(self, "MODEL_NAME", None):
            raise NotImplementedError(
                f"{self.__class__.__name__} must define AGENT_NAME and MODEL_NAME constants."
            )

    @abstractmethod
    async def setup(self):
        pass

    @abstractmethod
    async def run(self):
        pass

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

    async def execute(self):
        await self.setup()
        metric = await self.run()
        return await self.record(metric)